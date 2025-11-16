// Function: sub_F92670
// Address: 0xf92670
//
__int64 __fastcall sub_F92670(__int64 ***a1, __int64 a2, _QWORD *a3, __int64 a4)
{
  __int64 v6; // r8
  __int64 v7; // r9
  __m128i v8; // xmm2
  __m128i v9; // xmm3
  __int64 v10; // rdx
  void (__fastcall *v11)(_BYTE *, _BYTE *, __int64); // rax
  __int64 v12; // rsi
  _QWORD *v13; // r12
  int v14; // eax
  unsigned __int64 v15; // rdi
  __int64 v16; // rsi
  _QWORD *v17; // rax
  __int64 v18; // rcx
  unsigned __int8 *v20; // r10
  __int64 **v21; // r11
  __int64 v22; // r13
  unsigned __int8 *v23; // rax
  unsigned __int8 *v24; // r13
  __int64 v25; // r13
  unsigned __int64 v26; // rdx
  __int64 v27; // rbx
  unsigned __int8 **v28; // rdi
  int v29; // ecx
  unsigned __int8 **v30; // r9
  unsigned __int64 v31; // rcx
  unsigned __int8 *v32; // rsi
  __int64 v33; // rax
  __int64 v34; // rbx
  int v35; // edx
  int v36; // r13d
  signed __int64 v37; // rax
  unsigned __int8 *v38; // [rsp+8h] [rbp-1A8h]
  __int64 **v39; // [rsp+18h] [rbp-198h]
  signed __int64 v40; // [rsp+28h] [rbp-188h]
  __int64 v42; // [rsp+38h] [rbp-178h]
  unsigned __int64 v43; // [rsp+48h] [rbp-168h]
  unsigned __int8 **v44; // [rsp+50h] [rbp-160h] BYREF
  __int64 v45; // [rsp+58h] [rbp-158h]
  _BYTE v46[32]; // [rsp+60h] [rbp-150h] BYREF
  __m128i v47; // [rsp+80h] [rbp-130h]
  __m128i v48; // [rsp+90h] [rbp-120h]
  _BYTE v49[16]; // [rsp+A0h] [rbp-110h] BYREF
  void (__fastcall *v50)(_BYTE *, _BYTE *, __int64); // [rsp+B0h] [rbp-100h]
  unsigned __int8 (__fastcall *v51)(_BYTE *, __int64, __int64); // [rsp+B8h] [rbp-F8h]
  __m128i v52; // [rsp+C0h] [rbp-F0h]
  __m128i v53; // [rsp+D0h] [rbp-E0h]
  _BYTE v54[16]; // [rsp+E0h] [rbp-D0h] BYREF
  void (__fastcall *v55)(_BYTE *, _BYTE *, __int64); // [rsp+F0h] [rbp-C0h]
  __int64 v56; // [rsp+F8h] [rbp-B8h]
  __m128i v57; // [rsp+100h] [rbp-B0h] BYREF
  __m128i v58; // [rsp+110h] [rbp-A0h] BYREF
  _BYTE v59[16]; // [rsp+120h] [rbp-90h] BYREF
  void (__fastcall *v60)(_BYTE *, _BYTE *, __int64); // [rsp+130h] [rbp-80h]
  unsigned __int8 (__fastcall *v61)(_BYTE *, __int64, __int64); // [rsp+138h] [rbp-78h]
  __m128i v62; // [rsp+140h] [rbp-70h] BYREF
  __m128i v63; // [rsp+150h] [rbp-60h] BYREF
  _BYTE v64[16]; // [rsp+160h] [rbp-50h] BYREF
  void (__fastcall *v65)(_BYTE *, _BYTE *, __int64); // [rsp+170h] [rbp-40h]
  __int64 v66; // [rsp+178h] [rbp-38h]

  if ( !a2 )
    return 1;
  v40 = (unsigned int)qword_4F8D348;
  sub_AA72C0(&v57, a2, 0);
  v50 = 0;
  v47 = _mm_loadu_si128(&v57);
  v48 = _mm_loadu_si128(&v58);
  if ( v60 )
  {
    v60(v49, v59, 2);
    v51 = v61;
    v50 = v60;
  }
  v8 = _mm_loadu_si128(&v62);
  v9 = _mm_loadu_si128(&v63);
  v55 = 0;
  v52 = v8;
  v53 = v9;
  if ( !v65 )
  {
    v10 = v47.m128i_i64[0];
    v12 = v47.m128i_i64[0];
    if ( v47.m128i_i64[0] != v52.m128i_i64[0] )
      goto LABEL_6;
    goto LABEL_29;
  }
  v65(v54, v64, 2);
  v10 = v47.m128i_i64[0];
  v56 = v66;
  v11 = v65;
  v12 = v47.m128i_i64[0];
  v55 = v65;
  if ( v47.m128i_i64[0] == v52.m128i_i64[0] )
  {
LABEL_27:
    if ( v11 )
      v11(v54, v54, 3);
LABEL_29:
    if ( v50 )
      v50(v49, v49, 3);
    if ( v65 )
      v65(v64, v64, 3);
    if ( v60 )
      v60(v59, v59, 3);
    return 1;
  }
LABEL_6:
  v43 = 0;
  v42 = a4;
  v13 = (_QWORD *)((char *)a3 + ((8 * a4) & 0xFFFFFFFFFFFFFFE0LL));
  while ( 1 )
  {
    if ( !v12 )
      BUG();
    v14 = *(unsigned __int8 *)(v12 - 24);
    v15 = (unsigned int)(v14 - 30);
    if ( (unsigned int)v15 <= 0xA )
      goto LABEL_17;
    if ( (_BYTE)v14 == 62 )
    {
      v16 = v12 - 24;
      v17 = a3;
      v15 = (unsigned __int64)&a3[v42];
      v18 = (v42 * 8) >> 3;
      if ( (8 * a4) >> 5 > 0 )
      {
        while ( v16 != *v17 && v16 != v17[1] && v16 != v17[2] && v16 != v17[3] )
        {
          v17 += 4;
          if ( v13 == v17 )
          {
            v18 = (__int64)(v15 - (_QWORD)v13) >> 3;
            goto LABEL_57;
          }
        }
        goto LABEL_17;
      }
LABEL_57:
      if ( v18 != 2 )
      {
        if ( v18 != 3 )
        {
          if ( v18 != 1 )
          {
LABEL_60:
            if ( !v15 )
              break;
            goto LABEL_17;
          }
LABEL_74:
          if ( v16 == *v17 )
            goto LABEL_17;
          goto LABEL_60;
        }
        if ( v16 == *v17 )
          goto LABEL_17;
        ++v17;
      }
      if ( v16 == *v17 )
        goto LABEL_17;
      ++v17;
      goto LABEL_74;
    }
    if ( (unsigned int)(v14 - 42) > 0x11 && *(_BYTE *)(v12 - 24) != 63 )
      break;
    v20 = (unsigned __int8 *)(v12 - 24);
    v21 = *a1;
    v22 = 32LL * (*(_DWORD *)(v12 - 20) & 0x7FFFFFF);
    if ( (*(_BYTE *)(v12 - 17) & 0x40) != 0 )
    {
      v23 = *(unsigned __int8 **)(v12 - 32);
      v24 = &v23[v22];
    }
    else
    {
      v23 = &v20[-v22];
      v24 = (unsigned __int8 *)(v12 - 24);
    }
    v25 = v24 - v23;
    v44 = (unsigned __int8 **)v46;
    v26 = v25 >> 5;
    v45 = 0x400000000LL;
    v27 = v25 >> 5;
    if ( (unsigned __int64)v25 > 0x80 )
    {
      v38 = v23;
      v39 = v21;
      sub_C8D5F0((__int64)&v44, v46, v26, 8u, v6, v7);
      v30 = v44;
      v29 = v45;
      v26 = v25 >> 5;
      v21 = v39;
      v20 = (unsigned __int8 *)(v12 - 24);
      v23 = v38;
      v28 = &v44[(unsigned int)v45];
    }
    else
    {
      v28 = (unsigned __int8 **)v46;
      v29 = 0;
      v30 = (unsigned __int8 **)v46;
    }
    if ( v25 > 0 )
    {
      v31 = 0;
      do
      {
        v28[v31 / 8] = *(unsigned __int8 **)&v23[4 * v31];
        v31 += 8LL;
        --v27;
      }
      while ( v27 );
      v30 = v44;
      v29 = v45;
    }
    v32 = v20;
    LODWORD(v45) = v29 + v26;
    v33 = sub_DFCEF0(v21, v20, v30, (unsigned int)(v29 + v26), 3);
    v15 = (unsigned __int64)v44;
    v34 = v33;
    v36 = v35;
    if ( v44 != (unsigned __int8 **)v46 )
      _libc_free(v44, v32);
    if ( v36 == 1 )
      break;
    v37 = v34 + v43;
    if ( __OFADD__(v34, v43) )
    {
      if ( v34 > 0 )
        break;
      v43 = 0x8000000000000000LL;
    }
    else
    {
      v43 += v34;
      if ( v40 < v37 )
        break;
    }
    v10 = v47.m128i_i64[0];
LABEL_17:
    v12 = *(_QWORD *)(v10 + 8);
    v47.m128i_i16[4] = 0;
    v47.m128i_i64[0] = v12;
    v10 = v12;
    if ( v12 != v48.m128i_i64[0] )
    {
      while ( 1 )
      {
        if ( v12 )
          v12 -= 24;
        if ( !v50 )
          sub_4263D6(v15, v12, v10);
        v15 = (unsigned __int64)v49;
        if ( v51(v49, v12, v10) )
          break;
        v12 = *(_QWORD *)(v47.m128i_i64[0] + 8);
        v47.m128i_i16[4] = 0;
        v47.m128i_i64[0] = v12;
        v10 = v12;
        if ( v48.m128i_i64[0] == v12 )
          goto LABEL_25;
      }
      v12 = v47.m128i_i64[0];
      v10 = v47.m128i_i64[0];
    }
LABEL_25:
    if ( v52.m128i_i64[0] == v12 )
    {
      v11 = v55;
      goto LABEL_27;
    }
  }
  if ( v55 )
    v55(v54, v54, 3);
  if ( v50 )
    v50(v49, v49, 3);
  if ( v65 )
    v65(v64, v64, 3);
  if ( v60 )
    v60(v59, v59, 3);
  return 0;
}
