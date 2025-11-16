// Function: sub_377A7C0
// Address: 0x377a7c0
//
void __fastcall sub_377A7C0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __m128i a5)
{
  unsigned __int64 *v7; // rax
  __int64 v8; // rsi
  unsigned __int64 v9; // r15
  __int64 v10; // r12
  unsigned int v11; // r14d
  unsigned __int64 v12; // rcx
  int v13; // eax
  __int64 v14; // rdx
  __int64 v15; // rsi
  __int64 v16; // rdx
  unsigned __int16 *v17; // rax
  __int64 v18; // r8
  __int64 v19; // rsi
  __int64 v20; // r11
  unsigned int *v21; // r10
  int v22; // eax
  __int64 v23; // rdx
  __int16 v24; // cx
  __m128i v25; // xmm2
  __int64 v26; // rsi
  __int64 v27; // rax
  __int16 v28; // dx
  __int64 v29; // rax
  unsigned int v30; // r12d
  __int64 v31; // rsi
  __int16 *v32; // rax
  __int16 v33; // dx
  __int64 v34; // rax
  __int64 v35; // r9
  __m128i v36; // xmm0
  __int64 v37; // rsi
  __m128i v38; // kr00_16
  __int64 v39; // r8
  __m128i *v40; // rdx
  int v41; // ecx
  __int64 v42; // rax
  _QWORD *v43; // r12
  __m128i *v44; // r13
  __int64 v45; // r14
  _QWORD *v46; // rax
  __int64 v47; // rdx
  _QWORD *v48; // rax
  __int64 v49; // rdi
  int v50; // edx
  int v51; // r9d
  int v52; // edx
  int v53; // r9d
  unsigned __int8 *v54; // rax
  __m128i *v55; // rdi
  int v56; // edx
  __int64 v57; // rdx
  __int64 v58; // rax
  unsigned int v59; // edx
  unsigned int v60; // edx
  unsigned int v61; // eax
  __int64 v62; // rcx
  _QWORD *v63; // [rsp+0h] [rbp-150h]
  __int64 v64; // [rsp+8h] [rbp-148h]
  int v65; // [rsp+14h] [rbp-13Ch]
  __int128 *v66; // [rsp+20h] [rbp-130h]
  unsigned int v69; // [rsp+3Ch] [rbp-114h]
  __int64 v70; // [rsp+48h] [rbp-108h]
  _QWORD *v71; // [rsp+48h] [rbp-108h]
  __int64 v72; // [rsp+80h] [rbp-D0h] BYREF
  int v73; // [rsp+88h] [rbp-C8h]
  _QWORD *v74; // [rsp+90h] [rbp-C0h] BYREF
  __int64 v75; // [rsp+98h] [rbp-B8h]
  __int64 v76; // [rsp+A0h] [rbp-B0h] BYREF
  __int64 v77; // [rsp+A8h] [rbp-A8h]
  __int64 v78; // [rsp+B0h] [rbp-A0h] BYREF
  int v79; // [rsp+B8h] [rbp-98h]
  __m128i v80; // [rsp+C0h] [rbp-90h] BYREF
  __m128i v81; // [rsp+D0h] [rbp-80h] BYREF
  __int64 v82; // [rsp+E0h] [rbp-70h] BYREF
  __int64 v83; // [rsp+E8h] [rbp-68h]
  void *s[2]; // [rsp+F0h] [rbp-60h] BYREF
  __m128i v85; // [rsp+100h] [rbp-50h] BYREF

  v69 = *(_DWORD *)(a2 + 24);
  v7 = *(unsigned __int64 **)(a2 + 40);
  v8 = *(_QWORD *)(a2 + 80);
  v9 = *v7;
  v10 = v7[1];
  v72 = v8;
  v11 = *((_DWORD *)v7 + 2);
  v12 = v9;
  if ( v8 )
  {
    sub_B96E90((__int64)&v72, v8, 1);
    v12 = v9;
  }
  v13 = *(_DWORD *)(a2 + 72);
  v14 = a1[1];
  LODWORD(v75) = 0;
  LODWORD(v77) = 0;
  v15 = *a1;
  v73 = v13;
  v16 = *(_QWORD *)(v14 + 64);
  v17 = (unsigned __int16 *)(*(_QWORD *)(v12 + 48) + 16LL * v11);
  v74 = 0;
  v18 = *((_QWORD *)v17 + 1);
  v76 = 0;
  sub_2FE6CC0((__int64)s, v15, v16, *v17, v18);
  if ( LOBYTE(s[0]) == 6 )
  {
    v26 = v9;
    sub_375E8D0((__int64)a1, v9, v10, (__int64)&v74, (__int64)&v76);
  }
  else
  {
    v19 = *(_QWORD *)(a2 + 80);
    v20 = a1[1];
    v78 = v19;
    if ( v19 )
    {
      v70 = v20;
      sub_B96E90((__int64)&v78, v19, 1);
      v20 = v70;
    }
    v21 = *(unsigned int **)(a2 + 40);
    v22 = *(_DWORD *)(a2 + 72);
    v81.m128i_i16[0] = 0;
    v79 = v22;
    v80.m128i_i16[0] = 0;
    v80.m128i_i64[1] = 0;
    v81.m128i_i64[1] = 0;
    v66 = (__int128 *)v21;
    v71 = (_QWORD *)v20;
    v23 = *(_QWORD *)(*(_QWORD *)v21 + 48LL) + 16LL * v21[2];
    v24 = *(_WORD *)v23;
    v83 = *(_QWORD *)(v23 + 8);
    LOWORD(v82) = v24;
    sub_33D0340((__int64)s, v20, &v82);
    v25 = _mm_loadu_si128(&v85);
    v80 = _mm_loadu_si128((const __m128i *)s);
    v81 = v25;
    sub_3408290((__int64)s, v71, v66, (__int64)&v78, (unsigned int *)&v80, (unsigned int *)&v81, a5);
    v26 = v78;
    if ( v78 )
      sub_B91220((__int64)&v78, v78);
    v74 = s[0];
    LODWORD(v75) = s[1];
    v76 = v85.m128i_i64[0];
    LODWORD(v77) = v85.m128i_i32[2];
  }
  v27 = v74[6] + 16LL * (unsigned int)v75;
  v28 = *(_WORD *)v27;
  v29 = *(_QWORD *)(v27 + 8);
  v80.m128i_i16[0] = v28;
  v80.m128i_i64[1] = v29;
  if ( v28 )
  {
    if ( (unsigned __int16)(v28 - 176) > 0x34u )
    {
LABEL_11:
      v30 = word_4456340[v80.m128i_u16[0] - 1];
      goto LABEL_14;
    }
  }
  else if ( !sub_3007100((__int64)&v80) )
  {
    goto LABEL_13;
  }
  sub_CA17B0(
    "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::ge"
    "tVectorElementCount() instead");
  if ( v80.m128i_i16[0] )
  {
    if ( (unsigned __int16)(v80.m128i_i16[0] - 176) <= 0x34u )
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
    goto LABEL_11;
  }
LABEL_13:
  v30 = sub_3007130((__int64)&v80, v26);
LABEL_14:
  v81.m128i_i64[1] = 0;
  v31 = a1[1];
  v81.m128i_i16[0] = 0;
  v32 = *(__int16 **)(a2 + 48);
  v33 = *v32;
  v34 = *((_QWORD *)v32 + 1);
  LOWORD(v82) = v33;
  v83 = v34;
  sub_33D0340((__int64)s, v31, &v82);
  v36 = _mm_loadu_si128((const __m128i *)s);
  v37 = v85.m128i_i64[0];
  v38 = v85;
  v81 = v36;
  if ( LOWORD(s[0]) )
  {
    if ( (unsigned __int16)(LOWORD(s[0]) - 176) > 0x34u )
    {
LABEL_16:
      v39 = word_4456340[v81.m128i_u16[0] - 1];
      goto LABEL_19;
    }
  }
  else if ( !sub_3007100((__int64)&v81) )
  {
    goto LABEL_18;
  }
  sub_CA17B0(
    "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::ge"
    "tVectorElementCount() instead");
  if ( v81.m128i_i16[0] )
  {
    if ( (unsigned __int16)(v81.m128i_i16[0] - 176) <= 0x34u )
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
    goto LABEL_16;
  }
LABEL_18:
  v39 = (unsigned int)sub_3007130((__int64)&v81, v37);
LABEL_19:
  s[0] = &v85;
  s[1] = (void *)0x800000000LL;
  if ( v30 > 8 )
  {
    v65 = v39;
    sub_C8D5F0((__int64)s, &v85, v30, 4u, v39, v35);
    memset(s[0], 255, 4LL * v30);
    LODWORD(s[1]) = v30;
    v40 = (__m128i *)s[0];
    LODWORD(v39) = v65;
  }
  else
  {
    if ( v30 )
    {
      v57 = 4LL * v30;
      if ( v57 )
      {
        if ( (unsigned int)v57 < 8 )
        {
          if ( ((4 * (_BYTE)v30) & 4) != 0 )
          {
            v85.m128i_i32[0] = -1;
            *(_DWORD *)((char *)&s[1] + (unsigned int)v57 + 4) = -1;
          }
          else if ( (_DWORD)v57 )
          {
            v85.m128i_i8[0] = -1;
          }
        }
        else
        {
          v58 = (unsigned int)v57;
          v59 = v57 - 1;
          *(void **)((char *)&s[1] + v58) = (void *)-1LL;
          if ( v59 >= 8 )
          {
            v60 = v59 & 0xFFFFFFF8;
            v61 = 0;
            do
            {
              v62 = v61;
              v61 += 8;
              *(__int64 *)((char *)v85.m128i_i64 + v62) = -1;
            }
            while ( v61 < v60 );
          }
        }
      }
    }
    LODWORD(s[1]) = v30;
    v40 = &v85;
  }
  if ( (_DWORD)v39 )
  {
    v41 = 2 * v39;
    v42 = 0;
    do
    {
      v40->m128i_i32[v42] = v39;
      LODWORD(v39) = v39 + 1;
      v40 = (__m128i *)s[0];
      ++v42;
    }
    while ( (_DWORD)v39 != v41 );
  }
  v43 = (_QWORD *)a1[1];
  v44 = v40;
  v82 = 0;
  LODWORD(v83) = 0;
  v45 = LODWORD(s[1]);
  v46 = sub_33F17F0(v43, 51, (__int64)&v82, v80.m128i_u32[0], v80.m128i_i64[1]);
  if ( v82 )
  {
    v63 = v46;
    v64 = v47;
    sub_B91220((__int64)&v82, v82);
    v46 = v63;
    v47 = v64;
  }
  v48 = sub_33FCE10(
          (__int64)v43,
          v80.m128i_u32[0],
          v80.m128i_i64[1],
          (__int64)&v72,
          (__int64)v74,
          v75,
          v36,
          (__int64)v46,
          v47,
          v44,
          v45);
  v49 = a1[1];
  v76 = (__int64)v48;
  LODWORD(v77) = v50;
  *(_QWORD *)a3 = sub_33FAF80(v49, v69, (__int64)&v72, v81.m128i_u32[0], v81.m128i_i64[1], v51, v36);
  *(_DWORD *)(a3 + 8) = v52;
  v54 = sub_33FAF80(a1[1], v69, (__int64)&v72, v38.m128i_i64[0], v38.m128i_i64[1], v53, v36);
  v55 = (__m128i *)s[0];
  *(_QWORD *)a4 = v54;
  *(_DWORD *)(a4 + 8) = v56;
  if ( v55 != &v85 )
    _libc_free((unsigned __int64)v55);
  if ( v72 )
    sub_B91220((__int64)&v72, v72);
}
