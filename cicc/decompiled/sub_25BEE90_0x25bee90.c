// Function: sub_25BEE90
// Address: 0x25bee90
//
__int64 __fastcall sub_25BEE90(_QWORD **a1, __int64 a2)
{
  unsigned __int64 v2; // rax
  unsigned int v3; // r12d
  unsigned __int8 *v4; // r13
  unsigned int v5; // eax
  unsigned __int16 v7; // ax
  __int64 v8; // r14
  char v9; // bl
  __int64 v10; // [rsp-90h] [rbp-90h] BYREF
  __m128i v11; // [rsp-88h] [rbp-88h] BYREF
  unsigned __int64 v12; // [rsp-78h] [rbp-78h]
  __int64 v13; // [rsp-70h] [rbp-70h]
  __int64 v14; // [rsp-68h] [rbp-68h]
  __int64 v15; // [rsp-60h] [rbp-60h]
  __int64 v16; // [rsp-58h] [rbp-58h]
  __int64 v17; // [rsp-50h] [rbp-50h]
  __int16 v18; // [rsp-48h] [rbp-48h]

  v2 = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v2 == a2 + 48 )
    goto LABEL_25;
  if ( !v2 )
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v2 - 24) - 30 > 0xA )
LABEL_25:
    BUG();
  v3 = 1;
  if ( *(_BYTE *)(v2 - 24) == 30 )
  {
    v4 = 0;
    if ( (*(_DWORD *)(v2 - 20) & 0x7FFFFFF) != 0 )
      v4 = *(unsigned __int8 **)(v2 - 32LL * (*(_DWORD *)(v2 - 20) & 0x7FFFFFF) - 24);
    LOBYTE(v5) = sub_98ED60(v4, 0, 0, 0, 0);
    v3 = v5;
    if ( !(_BYTE)v5 )
      return 0;
    if ( (unsigned __int8)sub_A74710(*a1, 0, 43) )
    {
      v11 = (__m128i)(unsigned __int64)a1[1];
      v12 = 0;
      v13 = 0;
      v14 = 0;
      v15 = 0;
      v16 = 0;
      v17 = 0;
      v18 = 257;
      if ( !(unsigned __int8)sub_9B6260((__int64)v4, &v11, 0) )
        return 0;
    }
    v7 = sub_A74820(*a1);
    if ( HIBYTE(v7) )
    {
      if ( (unsigned __int8)sub_BD5420(v4, (__int64)a1[1]) < (unsigned __int8)v7 )
        return 0;
    }
    v10 = sub_A747F0(*a1, 0, 97);
    if ( v10 )
    {
      v8 = sub_A72AA0(&v10);
      sub_99D930((__int64)&v11, v4, 0, 1u, 0, 0, 0, 0);
      v9 = sub_AB1BB0(v8, (__int64)&v11);
      if ( (unsigned int)v13 > 0x40 && v12 )
        j_j___libc_free_0_0(v12);
      if ( v11.m128i_i32[2] > 0x40u && v11.m128i_i64[0] )
        j_j___libc_free_0_0(v11.m128i_u64[0]);
      if ( !v9 )
        return 0;
    }
  }
  return v3;
}
