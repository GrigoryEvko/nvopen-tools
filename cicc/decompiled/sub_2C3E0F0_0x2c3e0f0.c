// Function: sub_2C3E0F0
// Address: 0x2c3e0f0
//
__int64 __fastcall sub_2C3E0F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rdx
  _BYTE *v8; // rdi
  const __m128i *v9; // rsi
  unsigned __int64 v10; // rdx
  __int64 v11; // r10
  const __m128i *v12; // rcx
  const __m128i *v13; // r8
  unsigned __int64 v14; // r13
  __int64 v15; // rax
  __m128i *v16; // rdi
  __m128i *v17; // rdx
  const __m128i *v18; // rax
  __m128i *v19; // rax
  __m128i *v20; // rax
  __int8 *v21; // rax
  __m128i *v22; // rax
  __m128i *v23; // rax
  __int8 *v24; // rax
  __m128i *v25; // rax
  __m128i *v26; // rax
  __int8 *v27; // rax
  __int64 v28; // r8
  __int64 v29; // r9
  const __m128i *v30; // r8
  __int64 v31; // rax
  __m128i *v32; // rdi
  __m128i *v33; // rdx
  const __m128i *v34; // rax
  __m128i *v35; // rax
  __m128i *v36; // rax
  __int8 *v37; // rax
  __m128i *v38; // rax
  __m128i *v39; // rdx
  __m128i *v40; // rdx
  __int8 *v41; // rdx
  __m128i *v42; // rdx
  __m128i *v43; // rax
  __m128i *v44; // rax
  __int8 *v45; // rax
  __m128i *v46; // rax
  __m128i *v47; // rax
  __int8 *v48; // rax
  __m128i *v49; // rax
  __m128i *v50; // rax
  __int8 *v51; // rax
  __int64 v53; // [rsp+10h] [rbp-360h]
  __int64 v54; // [rsp+20h] [rbp-350h]
  _BYTE *v55; // [rsp+20h] [rbp-350h]
  _BYTE v57[32]; // [rsp+40h] [rbp-330h] BYREF
  _BYTE v58[64]; // [rsp+60h] [rbp-310h] BYREF
  __m128i *v59; // [rsp+A0h] [rbp-2D0h]
  __m128i *v60; // [rsp+A8h] [rbp-2C8h]
  __int8 *v61; // [rsp+B0h] [rbp-2C0h]
  _BYTE v62[32]; // [rsp+C0h] [rbp-2B0h] BYREF
  _BYTE v63[64]; // [rsp+E0h] [rbp-290h] BYREF
  __m128i *v64; // [rsp+120h] [rbp-250h]
  __m128i *v65; // [rsp+128h] [rbp-248h]
  __int8 *v66; // [rsp+130h] [rbp-240h]
  _BYTE v67[32]; // [rsp+140h] [rbp-230h] BYREF
  _BYTE v68[64]; // [rsp+160h] [rbp-210h] BYREF
  __m128i *v69; // [rsp+1A0h] [rbp-1D0h]
  __m128i *v70; // [rsp+1A8h] [rbp-1C8h]
  __int8 *v71; // [rsp+1B0h] [rbp-1C0h]
  __int16 v72; // [rsp+1B8h] [rbp-1B8h]
  _BYTE v73[32]; // [rsp+1C0h] [rbp-1B0h] BYREF
  _BYTE v74[64]; // [rsp+1E0h] [rbp-190h] BYREF
  __m128i *v75; // [rsp+220h] [rbp-150h]
  __m128i *v76; // [rsp+228h] [rbp-148h]
  __int8 *v77; // [rsp+230h] [rbp-140h]
  __int16 v78; // [rsp+238h] [rbp-138h]
  _BYTE v79[32]; // [rsp+240h] [rbp-130h] BYREF
  _BYTE v80[64]; // [rsp+260h] [rbp-110h] BYREF
  __m128i *v81; // [rsp+2A0h] [rbp-D0h]
  __m128i *v82; // [rsp+2A8h] [rbp-C8h]
  __int8 *v83; // [rsp+2B0h] [rbp-C0h]
  __int16 v84; // [rsp+2B8h] [rbp-B8h]
  _BYTE v85[32]; // [rsp+2C0h] [rbp-B0h] BYREF
  _BYTE v86[64]; // [rsp+2E0h] [rbp-90h] BYREF
  __m128i *v87; // [rsp+320h] [rbp-50h]
  __m128i *v88; // [rsp+328h] [rbp-48h]
  __int8 *v89; // [rsp+330h] [rbp-40h]
  __int16 v90; // [rsp+338h] [rbp-38h]

  v6 = a2 + 120;
  v8 = v62;
  v9 = (const __m128i *)v63;
  sub_C8CD80((__int64)v62, (__int64)v63, v6, a4, a5, a6);
  v11 = a2;
  v64 = 0;
  v65 = 0;
  v12 = *(const __m128i **)(a2 + 224);
  v13 = *(const __m128i **)(a2 + 216);
  v66 = 0;
  v14 = (char *)v12 - (char *)v13;
  if ( v12 == v13 )
  {
    v16 = 0;
  }
  else
  {
    if ( v14 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_22;
    v15 = sub_22077B0((char *)v12 - (char *)v13);
    v11 = a2;
    v16 = (__m128i *)v15;
    v12 = *(const __m128i **)(a2 + 224);
    v13 = *(const __m128i **)(a2 + 216);
  }
  v64 = v16;
  v65 = v16;
  v66 = &v16->m128i_i8[v14];
  if ( v13 != v12 )
  {
    v17 = v16;
    v18 = v13;
    do
    {
      if ( v17 )
      {
        *v17 = _mm_loadu_si128(v18);
        v17[1].m128i_i64[0] = v18[1].m128i_i64[0];
      }
      v18 = (const __m128i *)((char *)v18 + 24);
      v17 = (__m128i *)((char *)v17 + 24);
    }
    while ( v18 != v12 );
    v16 = (__m128i *)((char *)v16 + 8 * ((unsigned __int64)((char *)&v18[-2].m128i_u64[1] - (char *)v13) >> 3) + 24);
  }
  v65 = v16;
  v54 = v11;
  sub_C8CF70((__int64)v79, v80, 8, (__int64)v63, (__int64)v62);
  v19 = v64;
  v64 = 0;
  v81 = v19;
  v20 = v65;
  v65 = 0;
  v82 = v20;
  v21 = v66;
  v66 = 0;
  v83 = v21;
  sub_C8CF70((__int64)v85, v86, 8, (__int64)v80, (__int64)v79);
  v22 = v81;
  v81 = 0;
  v87 = v22;
  v23 = v82;
  v82 = 0;
  v88 = v23;
  v24 = v83;
  v83 = 0;
  v89 = v24;
  sub_C8CF70((__int64)v73, v74, 8, (__int64)v86, (__int64)v85);
  v25 = v87;
  v87 = 0;
  v75 = v25;
  v26 = v88;
  v88 = 0;
  v76 = v26;
  v27 = v89;
  v89 = 0;
  v77 = v27;
  sub_2AB1B10((__int64)v85);
  v78 = 256;
  sub_2AB1B10((__int64)v79);
  v8 = v57;
  v53 = v54;
  sub_C8CD80((__int64)v57, (__int64)v58, v54, (__int64)v58, v28, v29);
  v59 = 0;
  v60 = 0;
  v9 = *(const __m128i **)(v54 + 104);
  v30 = *(const __m128i **)(v54 + 96);
  v61 = 0;
  v10 = (char *)v9 - (char *)v30;
  if ( v9 != v30 )
  {
    if ( v10 <= 0x7FFFFFFFFFFFFFF8LL )
    {
      v55 = (_BYTE *)((char *)v9 - (char *)v30);
      v31 = sub_22077B0((char *)v9 - (char *)v30);
      v10 = (unsigned __int64)v55;
      v32 = (__m128i *)v31;
      v9 = *(const __m128i **)(v53 + 104);
      v30 = *(const __m128i **)(v53 + 96);
      goto LABEL_13;
    }
LABEL_22:
    sub_4261EA(v8, v9, v10);
  }
  v32 = 0;
LABEL_13:
  v59 = v32;
  v60 = v32;
  v61 = &v32->m128i_i8[v10];
  if ( v9 != v30 )
  {
    v33 = v32;
    v34 = v30;
    do
    {
      if ( v33 )
      {
        *v33 = _mm_loadu_si128(v34);
        v33[1].m128i_i64[0] = v34[1].m128i_i64[0];
      }
      v34 = (const __m128i *)((char *)v34 + 24);
      v33 = (__m128i *)((char *)v33 + 24);
    }
    while ( v9 != v34 );
    v32 = (__m128i *)((char *)v32 + 8 * ((unsigned __int64)((char *)&v9[-2].m128i_u64[1] - (char *)v30) >> 3) + 24);
  }
  v60 = v32;
  sub_C8CF70((__int64)v79, v80, 8, (__int64)v58, (__int64)v57);
  v35 = v59;
  v59 = 0;
  v81 = v35;
  v36 = v60;
  v60 = 0;
  v82 = v36;
  v37 = v61;
  v61 = 0;
  v83 = v37;
  sub_C8CF70((__int64)v85, v86, 8, (__int64)v80, (__int64)v79);
  v38 = v81;
  v81 = 0;
  v87 = v38;
  v88 = v82;
  v82 = 0;
  v89 = v83;
  v83 = 0;
  sub_C8CF70((__int64)v67, v68, 8, (__int64)v86, (__int64)v85);
  v39 = v87;
  v87 = 0;
  v69 = v39;
  v40 = v88;
  v88 = 0;
  v70 = v40;
  v41 = v89;
  v89 = 0;
  v71 = v41;
  sub_2AB1B10((__int64)v85);
  v72 = 256;
  sub_2AB1B10((__int64)v79);
  sub_C8CF70((__int64)v85, v86, 8, (__int64)v74, (__int64)v73);
  v42 = v75;
  v75 = 0;
  v87 = v42;
  v88 = v76;
  v76 = 0;
  v89 = v77;
  v77 = 0;
  v90 = v78;
  sub_C8CF70((__int64)v79, v80, 8, (__int64)v68, (__int64)v67);
  v43 = v69;
  v69 = 0;
  v81 = v43;
  v44 = v70;
  v70 = 0;
  v82 = v44;
  v45 = v71;
  v71 = 0;
  v83 = v45;
  v84 = v72;
  sub_C8CF70(a1, (void *)(a1 + 32), 8, (__int64)v80, (__int64)v79);
  v46 = v81;
  v81 = 0;
  *(_QWORD *)(a1 + 96) = v46;
  v47 = v82;
  v82 = 0;
  *(_QWORD *)(a1 + 104) = v47;
  v48 = v83;
  v83 = 0;
  *(_QWORD *)(a1 + 112) = v48;
  *(_WORD *)(a1 + 120) = v84;
  sub_C8CF70(a1 + 128, (void *)(a1 + 160), 8, (__int64)v86, (__int64)v85);
  v49 = v87;
  v87 = 0;
  *(_QWORD *)(a1 + 224) = v49;
  v50 = v88;
  v88 = 0;
  *(_QWORD *)(a1 + 232) = v50;
  v51 = v89;
  v89 = 0;
  *(_QWORD *)(a1 + 240) = v51;
  *(_WORD *)(a1 + 248) = v90;
  sub_2AB1B10((__int64)v79);
  sub_2AB1B10((__int64)v85);
  sub_2AB1B10((__int64)v67);
  sub_2AB1B10((__int64)v57);
  sub_2AB1B10((__int64)v73);
  sub_2AB1B10((__int64)v62);
  return a1;
}
