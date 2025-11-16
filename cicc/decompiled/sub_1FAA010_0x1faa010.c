// Function: sub_1FAA010
// Address: 0x1faa010
//
__int64 __fastcall sub_1FAA010(__int64 *a1, __int64 a2, double a3, double a4, __m128i a5)
{
  __int64 v7; // rax
  __int64 v8; // rcx
  __m128i v9; // xmm0
  __int64 v10; // r12
  unsigned __int64 v11; // r13
  __int64 v12; // r15
  __int64 v13; // rax
  unsigned __int8 v14; // r8
  const void **v15; // rax
  int v16; // eax
  __int64 v17; // r9
  int v18; // eax
  __int64 v19; // r10
  __int64 v20; // rax
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // rax
  __int64 v24; // rdi
  unsigned __int16 v25; // r9
  __int64 v26; // r10
  char v27; // al
  unsigned int v28; // r15d
  __int64 v29; // rsi
  __int64 *v30; // r10
  __int64 result; // rax
  bool v32; // al
  __int64 v33; // rsi
  int v34; // esi
  __int64 v35; // rsi
  __int64 v36; // rsi
  __int64 *v37; // r15
  __int64 v38; // rsi
  char v39; // al
  __int128 v40; // [rsp-10h] [rbp-90h]
  unsigned __int16 v41; // [rsp+Ch] [rbp-74h]
  unsigned __int16 v42; // [rsp+Ch] [rbp-74h]
  unsigned __int8 v43; // [rsp+10h] [rbp-70h]
  __int64 v44; // [rsp+10h] [rbp-70h]
  unsigned __int8 v45; // [rsp+10h] [rbp-70h]
  __int64 v46; // [rsp+10h] [rbp-70h]
  __int64 v47; // [rsp+18h] [rbp-68h]
  unsigned __int8 v48; // [rsp+18h] [rbp-68h]
  __int64 *v49; // [rsp+18h] [rbp-68h]
  __int64 v50; // [rsp+18h] [rbp-68h]
  __int64 v51; // [rsp+18h] [rbp-68h]
  __int64 v52; // [rsp+18h] [rbp-68h]
  unsigned __int8 v53; // [rsp+18h] [rbp-68h]
  __int64 v54; // [rsp+20h] [rbp-60h]
  __int64 v55; // [rsp+20h] [rbp-60h]
  unsigned int v56; // [rsp+30h] [rbp-50h] BYREF
  const void **v57; // [rsp+38h] [rbp-48h]
  __int64 v58; // [rsp+40h] [rbp-40h] BYREF
  int v59; // [rsp+48h] [rbp-38h]

  v7 = *(_QWORD *)(a2 + 32);
  v8 = *(_QWORD *)v7;
  v9 = _mm_loadu_si128((const __m128i *)(v7 + 40));
  v10 = *(_QWORD *)v7;
  v11 = *(_QWORD *)(v7 + 8);
  v12 = *(_QWORD *)(v7 + 40);
  v13 = *(_QWORD *)(*(_QWORD *)v7 + 40LL) + 16LL * *(unsigned int *)(v7 + 8);
  v14 = *(_BYTE *)v13;
  v15 = *(const void ***)(v13 + 8);
  LOBYTE(v56) = v14;
  v57 = v15;
  if ( v14 )
  {
    if ( (unsigned __int8)(v14 - 14) > 0x5Fu )
      goto LABEL_3;
  }
  else
  {
    v50 = v8;
    v32 = sub_1F58D20((__int64)&v56);
    v8 = v50;
    v14 = 0;
    if ( !v32 )
      goto LABEL_3;
  }
  v45 = v14;
  v51 = v8;
  result = (__int64)sub_1FA8C50((__int64)a1, a2, *(double *)v9.m128i_i64, a4, a5);
  v8 = v51;
  v14 = v45;
  if ( result )
    return result;
LABEL_3:
  v16 = *(unsigned __int16 *)(v8 + 24);
  if ( v16 == 32 || (v17 = 0, v16 == 10) )
  {
    v17 = 0;
    if ( (*(_BYTE *)(v8 + 26) & 8) == 0 )
      v17 = v8;
  }
  v18 = *(unsigned __int16 *)(v12 + 24);
  v19 = *a1;
  if ( (v18 == 32 || v18 == 10) && (*(_BYTE *)(v12 + 26) & 8) == 0 && v17 )
  {
    v33 = *(_QWORD *)(a2 + 72);
    v58 = v33;
    if ( v33 )
    {
      v52 = v19;
      v54 = v17;
      sub_1623A60((__int64)&v58, v33, 2);
      v19 = v52;
      v17 = v54;
    }
    v34 = *(unsigned __int16 *)(a2 + 24);
    v59 = *(_DWORD *)(a2 + 64);
    result = sub_1D392A0(v19, v34, (__int64)&v58, v56, v57, v17, v9, a4, a5, v12);
    v35 = v58;
    if ( v58 )
    {
LABEL_29:
      v55 = result;
      sub_161E7C0((__int64)&v58, v35);
      return v55;
    }
  }
  else
  {
    v43 = v14;
    v47 = v8;
    v20 = sub_1D23600(*a1, v10);
    v21 = v47;
    v22 = v43;
    if ( v20 && (v23 = sub_1D23600(*a1, v9.m128i_i64[0]), v21 = v47, v22 = v43, !v23) )
    {
      v36 = *(_QWORD *)(a2 + 72);
      v37 = (__int64 *)*a1;
      v58 = v36;
      if ( v36 )
        sub_1623A60((__int64)&v58, v36, 2);
      *((_QWORD *)&v40 + 1) = v11;
      v38 = *(unsigned __int16 *)(a2 + 24);
      *(_QWORD *)&v40 = v10;
      v59 = *(_DWORD *)(a2 + 64);
      result = (__int64)sub_1D332F0(
                          v37,
                          v38,
                          (__int64)&v58,
                          v56,
                          v57,
                          0,
                          *(double *)v9.m128i_i64,
                          a4,
                          a5,
                          v9.m128i_i64[0],
                          v9.m128i_u64[1],
                          v40);
    }
    else
    {
      v24 = *a1;
      v25 = *(_WORD *)(a2 + 24);
      v26 = *(_QWORD *)(*a1 + 16);
      if ( ((_BYTE)v22 == 1 || (_BYTE)v22 && *(_QWORD *)(v26 + 8LL * (unsigned __int8)v22 + 120))
        && v25 <= 0x102u
        && !*(_BYTE *)(v25 + v26 + 259LL * (unsigned __int8)v22 + 2422) )
      {
        return 0;
      }
      if ( *(_WORD *)(v21 + 24) != 48 )
      {
        v41 = *(_WORD *)(a2 + 24);
        v44 = *(_QWORD *)(*a1 + 16);
        v48 = v22;
        v27 = sub_1D1F9F0(v24, v10, v11, 0);
        v22 = v48;
        v26 = v44;
        v25 = v41;
        if ( !v27 )
          return 0;
      }
      if ( *(_WORD *)(v12 + 24) != 48 )
      {
        v42 = v25;
        v46 = v26;
        v53 = v22;
        v39 = sub_1D1F9F0(*a1, v9.m128i_i64[0], v9.m128i_i64[1], 0);
        v22 = v53;
        v26 = v46;
        v25 = v42;
        if ( !v39 )
          return 0;
      }
      if ( (_BYTE)v22 != 1 && (!(_BYTE)v22 || !*(_QWORD *)(v26 + 8LL * (unsigned __int8)v22 + 120)) )
        return 0;
      v28 = *(_DWORD *)&aT_2[4 * (unsigned __int16)(v25 - 114)];
      if ( v28 > 0x102 || *(_BYTE *)(v28 + 259 * v22 + v26 + 2422) )
        return 0;
      v29 = *(_QWORD *)(a2 + 72);
      v30 = (__int64 *)*a1;
      v58 = v29;
      if ( v29 )
      {
        v49 = v30;
        sub_1623A60((__int64)&v58, v29, 2);
        v30 = v49;
      }
      v59 = *(_DWORD *)(a2 + 64);
      result = (__int64)sub_1D332F0(
                          v30,
                          v28,
                          (__int64)&v58,
                          v56,
                          v57,
                          0,
                          *(double *)v9.m128i_i64,
                          a4,
                          a5,
                          v10,
                          v11,
                          *(_OWORD *)&v9);
    }
    v35 = v58;
    if ( v58 )
      goto LABEL_29;
  }
  return result;
}
