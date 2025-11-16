// Function: sub_3260C60
// Address: 0x3260c60
//
char *__fastcall sub_3260C60(unsigned int *src, unsigned int *a2, char *a3, __int64 a4)
{
  __int64 v5; // rax
  unsigned int *v6; // r15
  __int64 v7; // r13
  __int64 v9; // r14
  __int64 v10; // rbx
  unsigned int *v11; // rdi
  __int64 v12; // rsi

  v5 = ((char *)a2 - (char *)src) >> 4;
  v6 = src;
  v7 = 2 * a4;
  if ( 2 * a4 <= v5 )
  {
    v9 = 32 * a4;
    v10 = -4 * a4;
    do
    {
      v11 = v6;
      v6 = (unsigned int *)((char *)v6 + v9);
      a3 = sub_3260A10(v11, &v6[v10], &v6[v10], v6, a3);
      v5 = ((char *)a2 - (char *)v6) >> 4;
    }
    while ( v5 >= v7 );
  }
  v12 = a4;
  if ( v5 <= a4 )
    v12 = v5;
  return sub_3260A10(v6, &v6[4 * v12], &v6[4 * v12], a2, a3);
}
