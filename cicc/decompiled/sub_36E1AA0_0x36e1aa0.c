// Function: sub_36E1AA0
// Address: 0x36e1aa0
//
__int64 __fastcall sub_36E1AA0(__int64 a1, __int64 a2)
{
  unsigned int v2; // eax
  unsigned int v3; // r12d
  __int64 v5; // rdi
  __m128i v6; // [rsp+0h] [rbp-D0h] BYREF
  char v7[16]; // [rsp+10h] [rbp-C0h] BYREF
  __m128i *v8; // [rsp+20h] [rbp-B0h] BYREF
  __int16 v9; // [rsp+40h] [rbp-90h]
  __m128i v10; // [rsp+50h] [rbp-80h] BYREF
  char v11[112]; // [rsp+60h] [rbp-70h] BYREF

  v2 = sub_36E1940(a1 + 976, *(_BYTE *)(*(_QWORD *)(a2 + 112) + 36LL));
  if ( !v2 )
  {
    strcpy(v7, "Thread");
    v6.m128i_i64[0] = (__int64)v7;
    v6.m128i_i64[1] = 6;
    sub_35EF270(&v10, 1, "Atomics need scope > \"{}\".", &v6);
    v9 = 263;
    v8 = &v10;
    sub_C64D30((__int64)&v8, 1u);
  }
  v3 = v2;
  if ( v2 == 2 )
  {
    v5 = *(_QWORD *)(a1 + 1136);
    v10.m128i_i64[0] = (__int64)v11;
    strcpy(v11, "cluster scope");
    v10.m128i_i64[1] = 13;
    sub_305B5A0(v5, (__int64)&v10);
    if ( (char *)v10.m128i_i64[0] != v11 )
      j_j___libc_free_0(v10.m128i_u64[0]);
  }
  if ( (*(_BYTE *)(a2 + 32) & 8) != 0 )
    return 4;
  return v3;
}
