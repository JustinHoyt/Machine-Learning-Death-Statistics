
k=30 
t=0; 
x= D(find((D(:,1)==k) & (D(:,2)==t)),4);
plot(x) 
hold on
for t=1:9 
    x= D(find((D(:,1)==k) & (D(:,2)==t)),4);
    plot(x)
end

hold off

vals = [];
for t=0:9 
    x= D(find((D(:,1)==k) & (D(:,2)==t)),4);
    vals = [vals; x(end)];
end
mean(vals)

[v,T] = max(vals)

t=T(1)

x=D(find((D(:,1)==k) & (D(:,2)==t)),3);
x(end)


