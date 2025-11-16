// Function: sub_311E280
// Address: 0x311e280
//
char *__fastcall sub_311E280(unsigned __int64 **src, unsigned __int64 **a2, char *a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rax
  unsigned __int64 **v6; // r15
  __int64 v8; // r14
  __int64 v9; // rbx
  unsigned __int64 **v10; // rdi
  __int64 v13; // [rsp+8h] [rbp-38h]

  v5 = a2 - src;
  v6 = src;
  v13 = 2 * a4;
  if ( 2 * a4 <= v5 )
  {
    v8 = 16 * a4;
    v9 = -1 * a4;
    do
    {
      v10 = v6;
      v6 = (unsigned __int64 **)((char *)v6 + v8);
      a3 = sub_311E1B0(v10, &v6[v9], &v6[v9], v6, a3, a5);
      v5 = a2 - v6;
    }
    while ( v5 >= v13 );
  }
  if ( v5 > a4 )
    v5 = a4;
  return sub_311E1B0(v6, &v6[v5], &v6[v5], a2, a3, a5);
}
