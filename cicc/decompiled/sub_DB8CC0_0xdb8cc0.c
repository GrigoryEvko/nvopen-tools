// Function: sub_DB8CC0
// Address: 0xdb8cc0
//
__int64 __fastcall sub_DB8CC0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned __int8 a5,
        char a6,
        unsigned __int8 a7)
{
  __int64 *v8; // r12
  __int64 *v9; // rbx
  __int64 *v10; // rdi
  __int64 v12; // [rsp+0h] [rbp-1A0h] BYREF
  __int64 v13; // [rsp+8h] [rbp-198h]
  __int64 *v14; // [rsp+10h] [rbp-190h] BYREF
  unsigned int v15; // [rsp+18h] [rbp-188h]
  __int64 v16; // [rsp+170h] [rbp-30h] BYREF
  unsigned __int8 v17; // [rsp+178h] [rbp-28h]
  unsigned __int8 v18; // [rsp+179h] [rbp-27h]

  v8 = (__int64 *)&v14;
  v9 = (__int64 *)&v14;
  v12 = 0;
  v13 = 1;
  do
  {
    *v9 = -4;
    v9 += 11;
  }
  while ( v9 != &v16 );
  v17 = a5;
  v18 = a7;
  v16 = a3;
  sub_DB8AC0(a1, a2 * 8, (__int64)&v12, a3, a4, a5, a6, a7);
  if ( (v13 & 1) != 0 || (v8 = v14, a2 = 11LL * v15, v15) && (v9 = &v14[a2], &v14[a2] != v14) )
  {
    do
    {
      if ( *v8 != -4 && *v8 != -16 )
      {
        v10 = (__int64 *)v8[5];
        if ( v10 != v8 + 7 )
          _libc_free(v10, a2 * 8);
      }
      v8 += 11;
    }
    while ( v8 != v9 );
    if ( (v13 & 1) != 0 )
      return a1;
    v8 = v14;
    a2 = 11LL * v15;
  }
  sub_C7D6A0((__int64)v8, a2 * 8, 8);
  return a1;
}
