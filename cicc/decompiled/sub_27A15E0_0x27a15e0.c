// Function: sub_27A15E0
// Address: 0x27a15e0
//
char *__fastcall sub_27A15E0(int *a1, int *a2, char *a3, __int64 a4)
{
  __int64 v5; // rax
  __int64 v6; // r14
  int *v8; // rbx
  __int64 v9; // r15
  __int64 v10; // r13
  int *v11; // rdi
  __int64 v12; // rsi

  v5 = ((char *)a2 - (char *)a1) >> 5;
  v6 = 2 * a4;
  v8 = a1;
  if ( 2 * a4 <= v5 )
  {
    v9 = a4 << 6;
    v10 = -8 * a4;
    do
    {
      v11 = v8;
      v8 = (int *)((char *)v8 + v9);
      a3 = sub_27A14A0(v11, &v8[v10], &v8[v10], v8, a3);
      v5 = ((char *)a2 - (char *)v8) >> 5;
    }
    while ( v6 <= v5 );
  }
  v12 = a4;
  if ( a4 > v5 )
    v12 = v5;
  return sub_27A14A0(v8, &v8[8 * v12], &v8[8 * v12], a2, a3);
}
