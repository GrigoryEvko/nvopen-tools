// Function: sub_E6D310
// Address: 0xe6d310
//
unsigned __int64 __fastcall sub_E6D310(_QWORD *a1, _BYTE *a2, __int64 a3, char a4, __int64 a5, int a6)
{
  __int64 v6; // rax
  __m128i *v7; // rbx
  char v8; // dl
  char v9; // r14
  unsigned __int64 v10; // r13
  __int64 v12; // rax
  __int64 v13; // r14
  __int64 v14; // r10
  __int64 v15; // rax
  int v16; // [rsp+8h] [rbp-98h]
  __m128i *v20; // [rsp+20h] [rbp-80h] BYREF
  __int64 v21; // [rsp+28h] [rbp-78h]
  __m128i v22; // [rsp+30h] [rbp-70h] BYREF
  __m128i v23; // [rsp+40h] [rbp-60h] BYREF
  __m128i v24; // [rsp+50h] [rbp-50h] BYREF
  __int64 v25; // [rsp+60h] [rbp-40h]

  if ( a2 )
  {
    v20 = &v22;
    sub_E62BB0((__int64 *)&v20, a2, (__int64)&a2[a3]);
    v23.m128i_i64[0] = (__int64)&v24;
    v6 = v21;
    if ( v20 != &v22 )
    {
      v23.m128i_i64[0] = (__int64)v20;
      v24.m128i_i64[0] = v22.m128i_i64[0];
      goto LABEL_4;
    }
  }
  else
  {
    v22.m128i_i8[0] = 0;
    v6 = 0;
    v23.m128i_i64[0] = (__int64)&v24;
  }
  v24 = _mm_load_si128(&v22);
LABEL_4:
  v23.m128i_i64[1] = v6;
  v20 = &v22;
  v21 = 0;
  v22.m128i_i8[0] = 0;
  v25 = 0;
  v7 = sub_E6A210(a1 + 259, &v23);
  v9 = v8;
  if ( (__m128i *)v23.m128i_i64[0] != &v24 )
    j_j___libc_free_0(v23.m128i_i64[0], v24.m128i_i64[0] + 1);
  if ( v20 != &v22 )
    j_j___libc_free_0(v20, v22.m128i_i64[0] + 1);
  if ( !v9 )
    return v7[4].m128i_u64[0];
  v12 = a1[96];
  v13 = v7[2].m128i_i64[0];
  v14 = v7[2].m128i_i64[1];
  a1[106] += 168LL;
  v10 = (v12 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  if ( a1[97] >= v10 + 168 && v12 )
  {
    a1[96] = v10 + 168;
  }
  else
  {
    v16 = v14;
    v15 = sub_9D1E70((__int64)(a1 + 96), 168, 168, 3);
    LODWORD(v14) = v16;
    v10 = v15;
  }
  sub_E92760(v10, 2, v13, v14, (unsigned __int8)(a4 - 2) <= 1u, 0, 0);
  *(_QWORD *)v10 = &unk_49E1A10;
  *(_QWORD *)(v10 + 152) = a5;
  *(_DWORD *)(v10 + 160) = a6;
  v7[4].m128i_i64[0] = v10;
  sub_E6B260(a1, v10);
  return v10;
}
