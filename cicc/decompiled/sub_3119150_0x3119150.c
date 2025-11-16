// Function: sub_3119150
// Address: 0x3119150
//
unsigned __int64 *__fastcall sub_3119150(
        unsigned __int64 *a1,
        unsigned __int64 *a2,
        unsigned __int64 *a3,
        __int64 a4,
        __int64 a5)
{
  __int64 v5; // rax
  unsigned __int64 *v6; // r15
  __int64 v8; // r14
  __int64 v9; // rbx
  unsigned __int64 *v10; // rdi
  __int64 v13; // [rsp+8h] [rbp-38h]

  v5 = a2 - a1;
  v6 = a1;
  v13 = 2 * a4;
  if ( 2 * a4 <= v5 )
  {
    v8 = 16 * a4;
    v9 = -1 * a4;
    do
    {
      v10 = v6;
      v6 = (unsigned __int64 *)((char *)v6 + v8);
      a3 = sub_3118D30(v10, &v6[v9], &v6[v9], v6, a3, a5);
      v5 = a2 - v6;
    }
    while ( v5 >= v13 );
  }
  if ( v5 > a4 )
    v5 = a4;
  return sub_3118D30(v6, &v6[v5], &v6[v5], a2, a3, a5);
}
