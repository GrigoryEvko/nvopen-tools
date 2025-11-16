// Function: sub_2B1C0F0
// Address: 0x2b1c0f0
//
char *__fastcall sub_2B1C0F0(
        unsigned int *src,
        unsigned int *a2,
        char *a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 *a7)
{
  __int64 v8; // rax
  unsigned int *v9; // r15
  __int64 v10; // r13
  __int64 v11; // r14
  __int64 v12; // rbx
  unsigned int *v13; // rdi

  v8 = a2 - src;
  v9 = src;
  v10 = 2 * a4;
  if ( 2 * a4 <= v8 )
  {
    v11 = 8 * a4;
    v12 = -1 * a4;
    do
    {
      v13 = v9;
      v9 = (unsigned int *)((char *)v9 + v11);
      a3 = sub_2B1C020(v13, &v9[v12], &v9[v12], v9, a3, a6, a7);
      v8 = a2 - v9;
    }
    while ( v8 >= v10 );
  }
  if ( v8 > a4 )
    v8 = a4;
  return sub_2B1C020(v9, &v9[v8], &v9[v8], a2, a3, a6, a7);
}
