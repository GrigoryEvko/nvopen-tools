// Function: sub_1462C90
// Address: 0x1462c90
//
char *__fastcall sub_1462C90(
        __int64 **src,
        __int64 **a2,
        char *a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        _QWORD *a7,
        _QWORD *a8,
        __int64 *a9,
        __int64 a10)
{
  __int64 v11; // rax
  __int64 **v12; // r15
  __int64 v13; // r13
  __int64 v14; // r14
  __int64 v15; // rbx
  __int64 *v16; // rdi

  v11 = a2 - src;
  v12 = src;
  v13 = 2 * a4;
  if ( 2 * a4 <= v11 )
  {
    v14 = 16 * a4;
    v15 = -1 * a4;
    do
    {
      v16 = (__int64 *)v12;
      v12 = (__int64 **)((char *)v12 + v14);
      a3 = sub_1462B80(v16, (__int64 *)&v12[v15], &v12[v15], v12, a3, a6, a7, a8, a9, a10);
      v11 = a2 - v12;
    }
    while ( v11 >= v13 );
  }
  if ( v11 > a4 )
    v11 = a4;
  return sub_1462B80((__int64 *)v12, (__int64 *)&v12[v11], &v12[v11], a2, a3, a6, a7, a8, a9, a10);
}
