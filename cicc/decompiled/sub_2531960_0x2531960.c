// Function: sub_2531960
// Address: 0x2531960
//
void __fastcall sub_2531960(__int64 a1)
{
  __int64 *v1; // rdi
  __int64 v2; // [rsp+8h] [rbp-B8h] BYREF
  __m128i v3; // [rsp+10h] [rbp-B0h] BYREF
  __int64 v4; // [rsp+20h] [rbp-A0h] BYREF
  unsigned __int64 v5[2]; // [rsp+30h] [rbp-90h] BYREF
  _BYTE v6[16]; // [rsp+40h] [rbp-80h] BYREF
  const char *v7; // [rsp+50h] [rbp-70h] BYREF
  char v8; // [rsp+70h] [rbp-50h]
  char v9; // [rsp+71h] [rbp-4Fh]
  void *v10; // [rsp+80h] [rbp-40h] BYREF
  __int16 v11; // [rsp+A0h] [rbp-20h]

  v11 = 257;
  v2 = a1;
  v9 = 1;
  v7 = "Dependency Graph";
  v8 = 3;
  v5[0] = (unsigned __int64)v6;
  v5[1] = 0;
  v6[0] = 0;
  sub_2531550(&v3, (__int64)&v2, (void **)&v7, 0, &v10, (__int64)v5);
  if ( (_BYTE *)v5[0] != v6 )
    j_j___libc_free_0(v5[0]);
  v1 = (__int64 *)v3.m128i_i64[0];
  if ( v3.m128i_i64[1] )
  {
    sub_C67930(v3.m128i_i64[0], v3.m128i_i64[1], 0, 0);
    v1 = (__int64 *)v3.m128i_i64[0];
  }
  if ( v1 != &v4 )
    j_j___libc_free_0((unsigned __int64)v1);
}
