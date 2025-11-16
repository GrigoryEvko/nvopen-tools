// Function: sub_B8EAB0
// Address: 0xb8eab0
//
char *__fastcall sub_B8EAB0(int *a1, int *a2, int *a3, __int64 a4)
{
  int *v4; // r11
  __int64 v6; // rax
  __int64 v7; // r14
  __int64 v9; // r15
  __int64 v10; // r13

  v4 = a1;
  v6 = ((char *)a2 - (char *)a1) >> 4;
  v7 = 2 * a4;
  if ( 2 * a4 <= v6 )
  {
    v9 = 8 * a4;
    v10 = -4 * a4;
    do
    {
      a3 = (int *)sub_B8E860(v4, &v4[v9 + v10], &v4[v9 + v10], &v4[v9], a3);
      v6 = ((char *)a2 - (char *)v4) >> 4;
    }
    while ( v7 <= v6 );
  }
  if ( a4 <= v6 )
    v6 = a4;
  return sub_B8E860(v4, &v4[4 * v6], &v4[4 * v6], a2, a3);
}
