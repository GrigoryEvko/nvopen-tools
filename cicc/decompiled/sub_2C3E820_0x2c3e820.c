// Function: sub_2C3E820
// Address: 0x2c3e820
//
__int64 __fastcall sub_2C3E820(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _BYTE *v7; // rdi
  __m128i *v8; // rdx
  __int64 v9; // rcx
  const __m128i *v10; // rsi
  __int64 v11; // r9
  unsigned __int64 v12; // rbx
  __int64 v13; // rax
  __int64 v14; // r8
  const __m128i *v15; // rax
  __int16 v16; // ax
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // rcx
  __int64 v21; // r9
  unsigned __int64 v22; // rbx
  __int64 v23; // rax
  __int64 v24; // r8
  __m128i *v25; // rdx
  const __m128i *v26; // rax
  __int16 v27; // ax
  const __m128i *v28; // r9
  __int64 v29; // rax
  __m128i *v30; // r8
  __m128i *v31; // rdx
  const __m128i *v32; // rax
  __int16 v33; // ax
  __int64 v34; // rax
  __int64 v35; // rax
  unsigned __int64 v36; // rax
  __m128i *v37; // rax
  __m128i *v38; // rax
  __int8 *v39; // rax
  __int64 v40; // rax
  __int64 v41; // rax
  unsigned __int64 v42; // rax
  __m128i *v44; // [rsp+8h] [rbp-6D8h]
  _BYTE v46[32]; // [rsp+50h] [rbp-690h] BYREF
  _BYTE v47[64]; // [rsp+70h] [rbp-670h] BYREF
  __int64 v48; // [rsp+B0h] [rbp-630h]
  __int64 v49; // [rsp+B8h] [rbp-628h]
  unsigned __int64 v50; // [rsp+C0h] [rbp-620h]
  __int16 v51; // [rsp+C8h] [rbp-618h]
  _BYTE v52[32]; // [rsp+D0h] [rbp-610h] BYREF
  _BYTE v53[64]; // [rsp+F0h] [rbp-5F0h] BYREF
  __m128i *v54; // [rsp+130h] [rbp-5B0h]
  __m128i *v55; // [rsp+138h] [rbp-5A8h]
  __int8 *v56; // [rsp+140h] [rbp-5A0h]
  __int16 v57; // [rsp+148h] [rbp-598h]
  _BYTE v58[32]; // [rsp+160h] [rbp-580h] BYREF
  char v59[64]; // [rsp+180h] [rbp-560h] BYREF
  __int64 v60; // [rsp+1C0h] [rbp-520h]
  __int64 v61; // [rsp+1C8h] [rbp-518h]
  unsigned __int64 v62; // [rsp+1D0h] [rbp-510h]
  __int16 v63; // [rsp+1D8h] [rbp-508h]
  _QWORD v64[15]; // [rsp+1E0h] [rbp-500h] BYREF
  __int16 v65; // [rsp+258h] [rbp-488h]
  _BYTE v66[128]; // [rsp+270h] [rbp-470h] BYREF
  char v67[136]; // [rsp+2F0h] [rbp-3F0h] BYREF
  __int16 v68; // [rsp+378h] [rbp-368h]
  _BYTE v69[128]; // [rsp+380h] [rbp-360h] BYREF
  char v70[136]; // [rsp+400h] [rbp-2E0h] BYREF
  __int16 v71; // [rsp+488h] [rbp-258h]
  _BYTE v72[32]; // [rsp+490h] [rbp-250h] BYREF
  _BYTE v73[64]; // [rsp+4B0h] [rbp-230h] BYREF
  __int64 v74; // [rsp+4F0h] [rbp-1F0h]
  __int64 v75; // [rsp+4F8h] [rbp-1E8h]
  unsigned __int64 v76; // [rsp+500h] [rbp-1E0h]
  __int16 v77; // [rsp+508h] [rbp-1D8h]
  _BYTE v78[32]; // [rsp+510h] [rbp-1D0h] BYREF
  _BYTE v79[64]; // [rsp+530h] [rbp-1B0h] BYREF
  __m128i *v80; // [rsp+570h] [rbp-170h]
  __m128i *v81; // [rsp+578h] [rbp-168h]
  __int8 *v82; // [rsp+580h] [rbp-160h]
  __int16 v83; // [rsp+588h] [rbp-158h]
  __int16 v84; // [rsp+598h] [rbp-148h]
  _BYTE v85[32]; // [rsp+5A0h] [rbp-140h] BYREF
  char v86[64]; // [rsp+5C0h] [rbp-120h] BYREF
  __int64 v87; // [rsp+600h] [rbp-E0h]
  __int64 v88; // [rsp+608h] [rbp-D8h]
  unsigned __int64 v89; // [rsp+610h] [rbp-D0h]
  __int16 v90; // [rsp+618h] [rbp-C8h]
  _BYTE v91[32]; // [rsp+620h] [rbp-C0h] BYREF
  char v92[64]; // [rsp+640h] [rbp-A0h] BYREF
  __m128i *v93; // [rsp+680h] [rbp-60h]
  __m128i *v94; // [rsp+688h] [rbp-58h]
  __int8 *v95; // [rsp+690h] [rbp-50h]
  __int16 v96; // [rsp+698h] [rbp-48h]
  __int16 v97; // [rsp+6A8h] [rbp-38h]

  v7 = v58;
  sub_C8CD80((__int64)v58, (__int64)v59, a2 + 264, a4, a5, a6);
  v10 = *(const __m128i **)(a2 + 368);
  v11 = *(_QWORD *)(a2 + 360);
  v60 = 0;
  v61 = 0;
  v62 = 0;
  v12 = (unsigned __int64)v10 - v11;
  if ( v10 == (const __m128i *)v11 )
  {
    v14 = 0;
  }
  else
  {
    if ( v12 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_32;
    v13 = sub_22077B0((unsigned __int64)v10 - v11);
    v10 = *(const __m128i **)(a2 + 368);
    v11 = *(_QWORD *)(a2 + 360);
    v14 = v13;
  }
  v60 = v14;
  v61 = v14;
  v62 = v14 + v12;
  if ( v10 != (const __m128i *)v11 )
  {
    v8 = (__m128i *)v14;
    v15 = (const __m128i *)v11;
    do
    {
      if ( v8 )
      {
        *v8 = _mm_loadu_si128(v15);
        v9 = v15[1].m128i_i64[0];
        v8[1].m128i_i64[0] = v9;
      }
      v15 = (const __m128i *)((char *)v15 + 24);
      v8 = (__m128i *)((char *)v8 + 24);
    }
    while ( v15 != v10 );
    v14 += 8 * (((unsigned __int64)&v15[-2].m128i_u64[1] - v11) >> 3) + 24;
  }
  v16 = *(_WORD *)(a2 + 384);
  v61 = v14;
  v63 = v16;
  sub_2ABD910(v64, a2 + 392, (__int64)v8, v9, v14, v11);
  v65 = *(_WORD *)(a2 + 512);
  sub_2C2BB00((__int64)v72, (__int64)v58);
  sub_2C2BB00((__int64)v85, (__int64)v72);
  sub_2C2BB00((__int64)v69, (__int64)v85);
  sub_2AB1B10((__int64)v91);
  sub_2AB1B10((__int64)v85);
  v71 = 256;
  sub_2AB1B10((__int64)v78);
  sub_2AB1B10((__int64)v72);
  v7 = v46;
  sub_C8CD80((__int64)v46, (__int64)v47, a2, v17, v18, v19);
  v10 = *(const __m128i **)(a2 + 104);
  v21 = *(_QWORD *)(a2 + 96);
  v48 = 0;
  v49 = 0;
  v50 = 0;
  v22 = (unsigned __int64)v10 - v21;
  if ( v10 == (const __m128i *)v21 )
  {
    v22 = 0;
    v24 = 0;
  }
  else
  {
    if ( v22 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_32;
    v23 = sub_22077B0((unsigned __int64)v10 - v21);
    v10 = *(const __m128i **)(a2 + 104);
    v21 = *(_QWORD *)(a2 + 96);
    v24 = v23;
  }
  v48 = v24;
  v49 = v24;
  v50 = v24 + v22;
  if ( (const __m128i *)v21 != v10 )
  {
    v25 = (__m128i *)v24;
    v26 = (const __m128i *)v21;
    do
    {
      if ( v25 )
      {
        *v25 = _mm_loadu_si128(v26);
        v20 = v26[1].m128i_i64[0];
        v25[1].m128i_i64[0] = v20;
      }
      v26 = (const __m128i *)((char *)v26 + 24);
      v25 = (__m128i *)((char *)v25 + 24);
    }
    while ( v26 != v10 );
    v24 += 8 * (((unsigned __int64)&v26[-2].m128i_u64[1] - v21) >> 3) + 24;
  }
  v27 = *(_WORD *)(a2 + 120);
  v49 = v24;
  v7 = v52;
  v51 = v27;
  sub_C8CD80((__int64)v52, (__int64)v53, a2 + 128, v20, v24, v21);
  v10 = *(const __m128i **)(a2 + 232);
  v28 = *(const __m128i **)(a2 + 224);
  v54 = 0;
  v55 = 0;
  v56 = 0;
  v8 = (__m128i *)((char *)v10 - (char *)v28);
  if ( v10 != v28 )
  {
    if ( (unsigned __int64)v8 <= 0x7FFFFFFFFFFFFFF8LL )
    {
      v44 = (__m128i *)((char *)v10 - (char *)v28);
      v29 = sub_22077B0((char *)v10 - (char *)v28);
      v10 = *(const __m128i **)(a2 + 232);
      v28 = *(const __m128i **)(a2 + 224);
      v8 = v44;
      v30 = (__m128i *)v29;
      goto LABEL_22;
    }
LABEL_32:
    sub_4261EA(v7, v10, v8);
  }
  v30 = 0;
LABEL_22:
  v54 = v30;
  v55 = v30;
  v56 = &v8->m128i_i8[(_QWORD)v30];
  if ( v28 != v10 )
  {
    v31 = v30;
    v32 = v28;
    do
    {
      if ( v31 )
      {
        *v31 = _mm_loadu_si128(v32);
        v31[1].m128i_i64[0] = v32[1].m128i_i64[0];
      }
      v32 = (const __m128i *)((char *)v32 + 24);
      v31 = (__m128i *)((char *)v31 + 24);
    }
    while ( v32 != v10 );
    v30 = (__m128i *)((char *)v30 + 8 * ((unsigned __int64)((char *)&v32[-2].m128i_u64[1] - (char *)v28) >> 3) + 24);
  }
  v33 = *(_WORD *)(a2 + 248);
  v55 = v30;
  v57 = v33;
  sub_C8CF70((__int64)v72, v73, 8, (__int64)v47, (__int64)v46);
  v34 = v48;
  v48 = 0;
  v74 = v34;
  v35 = v49;
  v49 = 0;
  v75 = v35;
  v36 = v50;
  v50 = 0;
  v76 = v36;
  v77 = v51;
  sub_C8CF70((__int64)v78, v79, 8, (__int64)v53, (__int64)v52);
  v37 = v54;
  v54 = 0;
  v80 = v37;
  v38 = v55;
  v55 = 0;
  v81 = v38;
  v39 = v56;
  v56 = 0;
  v82 = v39;
  v83 = v57;
  sub_C8CF70((__int64)v85, v86, 8, (__int64)v73, (__int64)v72);
  v40 = v74;
  v74 = 0;
  v87 = v40;
  v41 = v75;
  v75 = 0;
  v88 = v41;
  v42 = v76;
  v76 = 0;
  v89 = v42;
  v90 = v77;
  sub_C8CF70((__int64)v91, v92, 8, (__int64)v79, (__int64)v78);
  v93 = v80;
  v80 = 0;
  v94 = v81;
  v81 = 0;
  v95 = v82;
  v82 = 0;
  v96 = v83;
  sub_2C2BB00((__int64)v66, (__int64)v85);
  sub_2AB1B10((__int64)v91);
  sub_2AB1B10((__int64)v85);
  v68 = 256;
  sub_2AB1B10((__int64)v78);
  sub_2AB1B10((__int64)v72);
  sub_2C2BB00((__int64)v85, (__int64)v69);
  v97 = v71;
  sub_2C2BB00((__int64)v72, (__int64)v66);
  v84 = v68;
  sub_2C2BB00(a1, (__int64)v72);
  *(_WORD *)(a1 + 264) = v84;
  sub_2C2BB00(a1 + 272, (__int64)v85);
  *(_WORD *)(a1 + 536) = v97;
  sub_2AB1B10((__int64)v78);
  sub_2AB1B10((__int64)v72);
  sub_2AB1B10((__int64)v91);
  sub_2AB1B10((__int64)v85);
  sub_2AB1B10((__int64)v67);
  sub_2AB1B10((__int64)v66);
  sub_2AB1B10((__int64)v52);
  sub_2AB1B10((__int64)v46);
  sub_2AB1B10((__int64)v70);
  sub_2AB1B10((__int64)v69);
  sub_2AB1B10((__int64)v64);
  sub_2AB1B10((__int64)v58);
  return a1;
}
