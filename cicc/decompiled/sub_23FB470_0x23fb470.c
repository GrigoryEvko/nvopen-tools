// Function: sub_23FB470
// Address: 0x23fb470
//
char *__fastcall sub_23FB470(__int64 ***src, __int64 ***a2, char *a3, __int64 a4)
{
  __int64 v4; // rax
  __int64 ***v5; // r15
  __int64 v6; // r14
  __int64 v7; // rbx
  __int64 ***v8; // rdi
  __int64 v11; // [rsp+8h] [rbp-38h]

  v4 = a2 - src;
  v5 = src;
  v11 = 2 * a4;
  if ( 2 * a4 <= v4 )
  {
    v6 = 16 * a4;
    v7 = -1 * a4;
    do
    {
      v8 = v5;
      v5 = (__int64 ***)((char *)v5 + v6);
      a3 = sub_23FB390(v8, &v5[v7], &v5[v7], v5, a3);
      v4 = a2 - v5;
    }
    while ( v11 <= v4 );
  }
  if ( a4 <= v4 )
    v4 = a4;
  return sub_23FB390(v5, &v5[v4], &v5[v4], a2, a3);
}
