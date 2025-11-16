// Function: sub_1A2F3F0
// Address: 0x1a2f3f0
//
__int64 __fastcall sub_1A2F3F0(
        __int64 *a1,
        __int64 *a2,
        __int64 a3,
        double a4,
        double a5,
        double a6,
        __int64 a7,
        __int64 a8,
        __int64 a9,
        __int64 a10,
        __int64 a11,
        __int64 a12)
{
  __int64 *v13; // r12
  __int64 *v15; // r15
  __int64 v16; // rsi
  unsigned int v17; // r12d
  __int64 *v18; // rax
  __int64 *v19; // rcx
  _BYTE *v20; // rdi
  unsigned __int64 v21; // r8
  __int16 v22; // cx
  unsigned int v23; // r15d
  _QWORD *v24; // rax
  unsigned int v25; // r15d
  _QWORD *v26; // r12
  __int64 v27; // rax
  unsigned __int64 *v28; // rcx
  __m128i v29; // xmm0
  unsigned __int64 v30; // rsi
  __int64 v31; // rax
  __int64 v32; // rsi
  __int64 v33; // rsi
  __int64 v34; // rdx
  unsigned __int8 *v35; // rsi
  __int64 v36; // rdi
  __int64 *v38; // [rsp+10h] [rbp-A0h]
  __int64 *v39; // [rsp+18h] [rbp-98h]
  __int64 v40; // [rsp+18h] [rbp-98h]
  unsigned __int64 *v41; // [rsp+18h] [rbp-98h]
  __m128i v42; // [rsp+20h] [rbp-90h] BYREF
  __int64 v43; // [rsp+30h] [rbp-80h]
  __m128i v44; // [rsp+40h] [rbp-70h] BYREF
  __int64 v45; // [rsp+50h] [rbp-60h]
  __m128i v46; // [rsp+60h] [rbp-50h] BYREF
  __int16 v47; // [rsp+70h] [rbp-40h]

  v13 = a2;
  v15 = a1 + 24;
  if ( sub_127FA20(*a1, *a2) != *(_DWORD *)(a1[10] + 8) >> 8 )
  {
    v16 = a1[6];
    v17 = 1 << *(_WORD *)(v16 + 18);
    v39 = sub_1A1D0C0(v15, v16, "oldload");
    sub_15F8F50((__int64)v39, v17 >> 1);
    v18 = sub_1A1C950(*a1, v15, v39, a1[10]);
    v19 = *(__int64 **)(a3 - 48);
    v20 = (_BYTE *)*a1;
    v21 = a1[14] - a1[7];
    v46.m128i_i64[0] = (__int64)"insert";
    v47 = 259;
    v13 = sub_1A202F0(v20, (__int64)v15, (__int64)v18, v19, v21, &v46, a4, a5, a6);
  }
  v38 = sub_1A1C950(*a1, v15, v13, a1[9]);
  v22 = *(_WORD *)(a1[6] + 18);
  v40 = a1[6];
  LOWORD(v43) = 257;
  v23 = 1 << v22;
  v24 = sub_1648A60(64, 2u);
  v25 = v23 >> 1;
  v26 = v24;
  if ( v24 )
    sub_15F9650((__int64)v24, (__int64)v38, v40, 0, 0);
  v27 = a1[25];
  v28 = (unsigned __int64 *)a1[26];
  v29 = _mm_loadu_si128(&v42);
  v45 = v43;
  v44 = v29;
  v41 = v28;
  if ( v27 )
  {
    sub_157E9D0(v27 + 40, (__int64)v26);
    v30 = *v41;
    v31 = v26[3] & 7LL;
    v26[4] = v41;
    v30 &= 0xFFFFFFFFFFFFFFF8LL;
    v26[3] = v30 | v31;
    *(_QWORD *)(v30 + 8) = v26 + 3;
    *v41 = *v41 & 7 | (unsigned __int64)(v26 + 3);
  }
  sub_164B780((__int64)v26, v44.m128i_i64);
  v32 = a1[24];
  if ( v32 )
  {
    v46.m128i_i64[0] = a1[24];
    sub_1623A60((__int64)&v46, v32, 2);
    v33 = v26[6];
    v34 = (__int64)(v26 + 6);
    if ( v33 )
    {
      sub_161E7C0((__int64)(v26 + 6), v33);
      v34 = (__int64)(v26 + 6);
    }
    v35 = (unsigned __int8 *)v46.m128i_i64[0];
    v26[6] = v46.m128i_i64[0];
    if ( v35 )
      sub_1623210((__int64)&v46, v35, v34);
  }
  sub_15F9450((__int64)v26, v25);
  v46.m128i_i32[0] = 10;
  sub_15F4370((__int64)v26, a3, v46.m128i_i32, 1);
  if ( a10 || a11 || a12 )
    sub_1626170((__int64)v26, &a10);
  v36 = a1[4];
  v46.m128i_i64[0] = a3;
  sub_1A2EDE0(v36 + 208, v46.m128i_i64);
  return 1;
}
