// Function: sub_326C8E0
// Address: 0x326c8e0
//
__int64 __fastcall sub_326C8E0(__int64 *a1, __int64 a2)
{
  __int32 v2; // eax
  __int64 result; // rax
  unsigned __int16 *v6; // rdx
  char v7; // r13
  __int64 v8; // rsi
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // rcx
  int v12; // r14d
  __int64 *v13; // rdx
  int v14; // ecx
  __int64 v15; // rax
  bool v16; // zf
  __int64 v17; // roff
  __m128i v18; // xmm3
  int v19; // r9d
  char v20; // r15
  unsigned __int64 v21; // rdi
  int v22; // eax
  unsigned __int16 *v23; // rax
  __int64 v24; // rsi
  __int64 v25; // r15
  __m128i v26; // xmm1
  int v27; // r13d
  __int64 v28; // r8
  __int64 v29; // rcx
  __int64 v30; // rcx
  __int128 v31; // [rsp-128h] [rbp-128h]
  int v32; // [rsp-100h] [rbp-100h]
  __int64 v33; // [rsp-100h] [rbp-100h]
  __m128i v34; // [rsp-E8h] [rbp-E8h]
  __m128i v35; // [rsp-C8h] [rbp-C8h] BYREF
  __m128i v36; // [rsp-B8h] [rbp-B8h] BYREF
  __int64 v37; // [rsp-A8h] [rbp-A8h] BYREF
  int v38; // [rsp-A0h] [rbp-A0h]
  __int64 v39; // [rsp-98h] [rbp-98h] BYREF
  int v40; // [rsp-90h] [rbp-90h]
  __m128i v41; // [rsp-88h] [rbp-88h] BYREF
  __m128i v42; // [rsp-78h] [rbp-78h]
  char v43; // [rsp-64h] [rbp-64h]
  __int64 *v44; // [rsp-60h] [rbp-60h]
  unsigned __int64 v45; // [rsp-58h] [rbp-58h] BYREF
  unsigned int v46; // [rsp-50h] [rbp-50h]
  int v47; // [rsp-48h] [rbp-48h]
  char v48; // [rsp-44h] [rbp-44h]
  __int64 v49; // [rsp-8h] [rbp-8h] BYREF

  v2 = *(_DWORD *)(a2 + 24);
  if ( (unsigned int)(v2 - 191) > 1 )
    return 0;
  v6 = *(unsigned __int16 **)(a2 + 48);
  v7 = *((_BYTE *)a1 + 33);
  v8 = a1[1];
  v9 = *v6;
  if ( v2 == 192 )
  {
    if ( v7 )
    {
      v29 = 1;
      if ( (_WORD)v9 != 1 )
      {
        if ( !(_WORD)v9 )
          return 0;
        v29 = (unsigned __int16)v9;
        if ( !*(_QWORD *)(v8 + 8 * v9 + 112) )
          return 0;
      }
      if ( *(_BYTE *)(v8 + 500 * v29 + 6589) )
        return 0;
    }
    else
    {
      v30 = 1;
      if ( (_WORD)v9 != 1 )
      {
        if ( !(_WORD)v9 )
          return 0;
        v30 = (unsigned __int16)v9;
        if ( !*(_QWORD *)(v8 + 8 * v9 + 112) )
          return 0;
      }
      if ( (*(_BYTE *)(v8 + 500 * v30 + 6589) & 0xFB) != 0 )
        return 0;
      v7 = 1;
    }
    v12 = 175;
  }
  else if ( v7 )
  {
    v10 = 1;
    if ( (_WORD)v9 != 1 )
    {
      if ( !(_WORD)v9 )
        return 0;
      v10 = (unsigned __int16)v9;
      if ( !*(_QWORD *)(v8 + 8 * v9 + 112) )
        return 0;
    }
    if ( *(_BYTE *)(v8 + 500 * v10 + 6588) )
      return 0;
    v7 = 0;
    v12 = 174;
  }
  else
  {
    v11 = 1;
    if ( (_WORD)v9 != 1 )
    {
      if ( !(_WORD)v9 )
        return 0;
      v11 = (unsigned __int16)v9;
      if ( !*(_QWORD *)(v8 + 8 * v9 + 112) )
        return 0;
    }
    v12 = 174;
    if ( (*(_BYTE *)(v8 + 500 * v11 + 6588) & 0xFB) != 0 )
      return 0;
  }
  v41.m128i_i32[0] = v2;
  v13 = *(__int64 **)(a2 + 40);
  v42.m128i_i64[0] = (__int64)&v35;
  v42.m128i_i64[1] = (__int64)&v36;
  v38 = 0;
  v41.m128i_i32[2] = 56;
  v43 = 0;
  v44 = &v37;
  v46 = 64;
  v45 = 1;
  v48 = 0;
  v35.m128i_i32[2] = 0;
  v36.m128i_i32[2] = 0;
  v14 = *((_DWORD *)v13 + 2);
  v37 = 0;
  v35.m128i_i64[0] = 0;
  v36.m128i_i64[0] = 0;
  v15 = *v13;
  v16 = *(_DWORD *)(*v13 + 24) == 56;
  v37 = *v13;
  v38 = v14;
  if ( !v16 )
    return 0;
  v17 = *(_QWORD *)(v15 + 40);
  v34 = _mm_loadu_si128((const __m128i *)v17);
  v18 = _mm_loadu_si128((const __m128i *)(v17 + 40));
  v35.m128i_i64[0] = *(_QWORD *)v17;
  v35.m128i_i32[2] = v34.m128i_i32[2];
  v36.m128i_i64[0] = v18.m128i_i64[0];
  v36.m128i_i32[2] = v18.m128i_i32[2];
  v20 = sub_32657E0((__int64)&v45, v13[5]);
  if ( !v20 )
  {
    if ( v46 > 0x40 && v45 )
      j_j___libc_free_0_0(v45);
    return 0;
  }
  if ( !v48 )
  {
    if ( v46 <= 0x40 )
      goto LABEL_20;
    v21 = v45;
    if ( !v45 )
      goto LABEL_20;
    goto LABEL_18;
  }
  v20 = (v47 & *(_DWORD *)(a2 + 28)) == v47;
  if ( v46 > 0x40 )
  {
    v21 = v45;
    if ( v45 )
LABEL_18:
      j_j___libc_free_0_0(v21);
  }
  if ( !v20 )
    return 0;
LABEL_20:
  v22 = *(_DWORD *)(v37 + 28);
  if ( v7 )
  {
    if ( (v22 & 1) != 0 )
      goto LABEL_22;
    return 0;
  }
  if ( (v22 & 2) == 0 )
    return 0;
LABEL_22:
  v23 = *(unsigned __int16 **)(a2 + 48);
  v24 = *(_QWORD *)(a2 + 80);
  v25 = *a1;
  v26 = _mm_loadu_si128(&v36);
  v41 = _mm_loadu_si128(&v35);
  v42 = v26;
  v27 = *v23;
  v28 = *((_QWORD *)v23 + 1);
  v39 = v24;
  if ( v24 )
  {
    v32 = v28;
    sub_B96E90((__int64)&v39, v24, 1);
    LODWORD(v28) = v32;
  }
  *((_QWORD *)&v31 + 1) = 2;
  *(_QWORD *)&v31 = &v41;
  v40 = *(_DWORD *)(a2 + 72);
  result = sub_33FC220(v25, v12, (unsigned int)&v49 - 144, v27, v28, v19, v31);
  if ( v39 )
  {
    v33 = result;
    sub_B91220((__int64)&v39, v39);
    return v33;
  }
  return result;
}
