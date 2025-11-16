// Function: sub_378FB20
// Address: 0x378fb20
//
unsigned __int8 *__fastcall sub_378FB20(__int64 **a1, __int64 a2, __int64 a3, char a4)
{
  unsigned int v4; // r12d
  int v5; // r14d
  __int64 v7; // r9
  __int64 (__fastcall *v8)(__int64, __int64, unsigned int, __int64); // r11
  __int16 *v9; // rax
  unsigned __int16 v10; // si
  __int64 v11; // r8
  __int64 *v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  unsigned __int16 v17; // r13
  __int64 v18; // rax
  __int64 v19; // rsi
  __int128 v20; // xmm0
  __int64 v21; // rax
  __m128i v22; // xmm3
  _WORD *v23; // rdx
  unsigned int v24; // r15d
  __int16 v25; // ax
  __m128i **v26; // rax
  __m128i **v27; // rcx
  __m128i **i; // rdx
  __int64 v29; // rax
  __int64 v30; // rdx
  unsigned int v31; // r13d
  __int64 v32; // rdx
  __int16 v33; // ax
  __m128i *v34; // rax
  __int64 v35; // r8
  __int64 v36; // r9
  int v37; // edx
  int v38; // edi
  __m128i *v39; // rdx
  unsigned __int64 v40; // rax
  __m128i *v41; // r12
  __int64 v42; // rax
  __m128i **v43; // rax
  __int64 v44; // r12
  int v45; // edx
  __int64 v46; // rcx
  __m128i *v47; // rax
  int v48; // edx
  int v49; // edi
  __m128i *v50; // rdx
  unsigned __int64 v51; // rax
  unsigned __int64 v52; // r8
  __int64 m128i_i64; // r9
  __int64 v54; // rax
  __int64 *v55; // rax
  unsigned __int8 *v56; // r10
  __int64 v57; // rdx
  __int64 v58; // r11
  __int64 *v59; // rdi
  __int64 v60; // rax
  int v61; // ecx
  unsigned __int64 v62; // rsi
  __int64 v63; // rcx
  char v64; // r8
  __int16 v65; // ax
  __int64 v66; // rdx
  __int64 v67; // rdx
  __int64 *v68; // rdi
  __m128i *v69; // rbx
  __int64 v70; // r9
  int v71; // edx
  int v72; // r12d
  unsigned int v73; // edx
  __int64 v74; // rax
  __m128i **v75; // rax
  unsigned __int8 *v76; // r12
  int v78; // eax
  __int128 v79; // [rsp-10h] [rbp-2C0h]
  __int64 v80; // [rsp+0h] [rbp-2B0h]
  unsigned int v81; // [rsp+1Ch] [rbp-294h]
  __m128i v82; // [rsp+20h] [rbp-290h]
  unsigned int v83; // [rsp+40h] [rbp-270h]
  unsigned int v84; // [rsp+44h] [rbp-26Ch]
  _TBYTE v86; // [rsp+66h] [rbp-24Ah]
  __m128i v87; // [rsp+70h] [rbp-240h]
  unsigned __int64 v88; // [rsp+70h] [rbp-240h]
  unsigned __int64 v90; // [rsp+88h] [rbp-228h]
  __int16 v91; // [rsp+90h] [rbp-220h]
  __int64 v93; // [rsp+D0h] [rbp-1E0h] BYREF
  __int64 v94; // [rsp+D8h] [rbp-1D8h]
  unsigned __int16 v95; // [rsp+E0h] [rbp-1D0h] BYREF
  __int64 v96; // [rsp+E8h] [rbp-1C8h]
  __int64 v97; // [rsp+F0h] [rbp-1C0h] BYREF
  int v98; // [rsp+F8h] [rbp-1B8h]
  __int64 v99; // [rsp+100h] [rbp-1B0h] BYREF
  __int64 v100; // [rsp+108h] [rbp-1A8h]
  __int64 v101; // [rsp+110h] [rbp-1A0h]
  __int64 v102; // [rsp+118h] [rbp-198h]
  __int64 v103; // [rsp+120h] [rbp-190h] BYREF
  __int64 v104; // [rsp+128h] [rbp-188h]
  __int128 v105; // [rsp+130h] [rbp-180h] BYREF
  __int64 v106; // [rsp+140h] [rbp-170h]
  _OWORD v107[2]; // [rsp+150h] [rbp-160h] BYREF
  __m128i **v108; // [rsp+170h] [rbp-140h] BYREF
  __int64 v109; // [rsp+178h] [rbp-138h]
  _QWORD v110[38]; // [rsp+180h] [rbp-130h] BYREF

  v7 = (__int64)*a1;
  v8 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(**a1 + 592);
  v9 = *(__int16 **)(a3 + 48);
  v10 = *v9;
  v11 = *((_QWORD *)v9 + 1);
  v12 = a1[1];
  if ( v8 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v108, v7, v12[8], v10, v11);
    LOWORD(v93) = v109;
    v94 = v110[0];
  }
  else
  {
    LODWORD(v93) = v8(v7, v12[8], v10, v11);
    v94 = v13;
  }
  v17 = *(_WORD *)(a3 + 96);
  v18 = *(_QWORD *)(a3 + 104);
  v19 = *(_QWORD *)(a3 + 80);
  v95 = v17;
  v96 = v18;
  v97 = v19;
  if ( v19 )
  {
    sub_B96E90((__int64)&v97, v19, 1);
    v17 = v95;
  }
  v98 = *(_DWORD *)(a3 + 72);
  v20 = (__int128)_mm_loadu_si128((const __m128i *)*(_QWORD *)(a3 + 40));
  v21 = *(_QWORD *)(a3 + 112);
  v87 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a3 + 40) + 40LL));
  v22 = _mm_loadu_si128((const __m128i *)(v21 + 56));
  v91 = *(_WORD *)(v21 + 32);
  v107[0] = _mm_loadu_si128((const __m128i *)(v21 + 40));
  v107[1] = v22;
  if ( v17 )
  {
    if ( (unsigned __int16)(v17 - 176) > 0x34u )
      goto LABEL_7;
LABEL_37:
    sub_C64ED0("Generating widen scalable extending vector loads is not yet supported", 1u);
  }
  if ( sub_3007100((__int64)&v95) )
    goto LABEL_37;
LABEL_7:
  if ( (_WORD)v93 )
  {
    v23 = word_4456580;
    HIWORD(v86) = 0;
    *(_QWORD *)&v86 = (unsigned __int16)word_4456580[(unsigned __int16)v93 - 1];
  }
  else
  {
    v78 = sub_3009970((__int64)&v93, v19, v13, v14, v15);
    v17 = v95;
    HIWORD(v4) = HIWORD(v78);
    LOWORD(v86) = v78;
    *(_QWORD *)((char *)&v86 + 2) = v23;
  }
  LOWORD(v4) = LOWORD(v86);
  HIWORD(v24) = HIWORD(v4);
  if ( v17 )
  {
    v100 = 0;
    LOWORD(v99) = word_4456580[v17 - 1];
  }
  else
  {
    v65 = sub_3009970((__int64)&v95, v19, (__int64)v23, v14, v15);
    v17 = v95;
    LOWORD(v99) = v65;
    v100 = v66;
    if ( !v95 )
    {
      if ( !sub_3007100((__int64)&v95) )
        goto LABEL_41;
      goto LABEL_40;
    }
  }
  if ( (unsigned __int16)(v17 - 176) > 0x34u )
    goto LABEL_12;
LABEL_40:
  sub_CA17B0(
    "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::ge"
    "tVectorElementCount() instead");
  if ( v95 )
  {
    if ( (unsigned __int16)(v95 - 176) <= 0x34u )
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
LABEL_12:
    v81 = word_4456340[v95 - 1];
    v25 = v93;
    if ( (_WORD)v93 )
      goto LABEL_13;
LABEL_42:
    if ( !sub_3007100((__int64)&v93) )
      goto LABEL_43;
    goto LABEL_58;
  }
LABEL_41:
  v81 = sub_3007130((__int64)&v95, v19);
  v25 = v93;
  if ( !(_WORD)v93 )
    goto LABEL_42;
LABEL_13:
  if ( (unsigned __int16)(v25 - 176) > 0x34u )
  {
LABEL_14:
    v83 = word_4456340[(unsigned __int16)v93 - 1];
    goto LABEL_15;
  }
LABEL_58:
  sub_CA17B0(
    "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::ge"
    "tVectorElementCount() instead");
  if ( (_WORD)v93 )
  {
    if ( (unsigned __int16)(v93 - 176) <= 0x34u )
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
    goto LABEL_14;
  }
LABEL_43:
  v83 = sub_3007130((__int64)&v93, v19);
LABEL_15:
  v26 = (__m128i **)v110;
  v27 = (__m128i **)v110;
  v108 = (__m128i **)v110;
  v109 = 0x1000000000LL;
  if ( v83 )
  {
    if ( v83 > 0x10uLL )
    {
      sub_C8D5F0((__int64)&v108, v110, v83, 0x10u, v15, v16);
      v27 = v108;
      v26 = &v108[2 * (unsigned int)v109];
    }
    for ( i = &v27[2 * v83]; i != v26; v26 += 2 )
    {
      if ( v26 )
      {
        *v26 = 0;
        *((_DWORD *)v26 + 2) = 0;
      }
    }
    LODWORD(v109) = v83;
  }
  if ( (_WORD)v99 )
  {
    if ( (_WORD)v99 == 1 || (unsigned __int16)(v99 - 504) <= 7u )
      BUG();
    v30 = 16LL * ((unsigned __int16)v99 - 1);
    v29 = *(_QWORD *)&byte_444C4A0[v30];
    LOBYTE(v30) = byte_444C4A0[v30 + 8];
  }
  else
  {
    v29 = sub_3007260((__int64)&v99);
    v101 = v29;
    v102 = v30;
  }
  BYTE8(v105) = v30;
  *(_QWORD *)&v105 = v29;
  v84 = (unsigned __int64)sub_CA1930(&v105) >> 3;
  v31 = v84;
  v32 = *(_QWORD *)(a3 + 112);
  LOBYTE(v33) = *(_BYTE *)(v32 + 34);
  HIBYTE(v33) = 1;
  v34 = sub_33F1DB0(
          a1[1],
          a4,
          (__int64)&v97,
          v4,
          *(__int64 *)((char *)&v86 + 2),
          v33,
          v20,
          v87.m128i_i64[0],
          v87.m128i_i64[1],
          *(_OWORD *)v32,
          *(_QWORD *)(v32 + 16),
          v99,
          v100,
          v91,
          (__int64)v107);
  v38 = v37;
  v39 = v34;
  v40 = (unsigned __int64)v108;
  *v108 = v39;
  *(_DWORD *)(v40 + 8) = v38;
  v41 = *v108;
  v42 = *(unsigned int *)(a2 + 8);
  if ( v42 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
  {
    sub_C8D5F0(a2, (const void *)(a2 + 16), v42 + 1, 0x10u, v35, v36);
    v42 = *(unsigned int *)(a2 + 8);
  }
  v43 = (__m128i **)(*(_QWORD *)a2 + 16 * v42);
  *v43 = v41;
  v43[1] = (__m128i *)1;
  ++*(_DWORD *)(a2 + 8);
  if ( v81 <= 1 )
  {
    v81 = 1;
  }
  else
  {
    v44 = 2;
    v82 = v87;
    do
    {
      LOBYTE(v104) = 0;
      v103 = v31;
      v56 = sub_3409320(a1[1], v82.m128i_i64[0], v82.m128i_i64[1], v31, 0, (__int64)&v97, (__m128i)v20, 1);
      v58 = v57;
      v59 = a1[1];
      v60 = *(_QWORD *)(a3 + 112);
      LOBYTE(v5) = *(_BYTE *)(v60 + 34);
      v61 = v5;
      BYTE1(v61) = 1;
      v5 = v61;
      v62 = *(_QWORD *)v60 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v62 )
      {
        v63 = *(_QWORD *)(v60 + 8) + v31;
        v64 = *(_BYTE *)(v60 + 20);
        if ( (*(_QWORD *)v60 & 4) != 0 )
        {
          *((_QWORD *)&v105 + 1) = *(_QWORD *)(v60 + 8) + v31;
          BYTE4(v106) = v64;
          *(_QWORD *)&v105 = v62 | 4;
          LODWORD(v106) = *(_DWORD *)(v62 + 12);
        }
        else
        {
          *(_QWORD *)&v105 = *(_QWORD *)v60 & 0xFFFFFFFFFFFFFFF8LL;
          *((_QWORD *)&v105 + 1) = v63;
          BYTE4(v106) = v64;
          v67 = *(_QWORD *)(v62 + 8);
          if ( (unsigned int)*(unsigned __int8 *)(v67 + 8) - 17 <= 1 )
            v67 = **(_QWORD **)(v67 + 16);
          LODWORD(v106) = *(_DWORD *)(v67 + 8) >> 8;
        }
      }
      else
      {
        v45 = *(_DWORD *)(v60 + 16);
        v46 = *(_QWORD *)(v60 + 8) + v31;
        BYTE4(v106) = 0;
        *(_QWORD *)&v105 = 0;
        *((_QWORD *)&v105 + 1) = v46;
        LODWORD(v106) = v45;
      }
      LOWORD(v24) = LOWORD(v86);
      v47 = sub_33F1DB0(
              v59,
              a4,
              (__int64)&v97,
              v24,
              *(__int64 *)((char *)&v86 + 2),
              v5,
              v20,
              (__int64)v56,
              v58,
              v105,
              v106,
              v99,
              v100,
              v91,
              (__int64)v107);
      v49 = v48;
      v50 = v47;
      v51 = (unsigned __int64)v108;
      v52 = v90 & 0xFFFFFFFF00000000LL | 1;
      v108[v44] = v50;
      v90 = v52;
      *(_DWORD *)(v51 + v44 * 8 + 8) = v49;
      m128i_i64 = (__int64)v108[v44]->m128i_i64;
      v54 = *(unsigned int *)(a2 + 8);
      if ( v54 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
      {
        v80 = (__int64)v108[v44]->m128i_i64;
        v88 = v52;
        sub_C8D5F0(a2, (const void *)(a2 + 16), v54 + 1, 0x10u, v52, m128i_i64);
        v54 = *(unsigned int *)(a2 + 8);
        m128i_i64 = v80;
        v52 = v88;
      }
      v55 = (__int64 *)(*(_QWORD *)a2 + 16 * v54);
      v31 += v84;
      v44 += 2;
      *v55 = m128i_i64;
      v55[1] = v52;
      ++*(_DWORD *)(a2 + 8);
    }
    while ( v44 != 2LL * v81 );
  }
  LOWORD(v24) = LOWORD(v86);
  v103 = 0;
  v68 = a1[1];
  LODWORD(v104) = 0;
  v69 = (__m128i *)sub_33F17F0(v68, 51, (__int64)&v103, v24, *(__int64 *)((char *)&v86 + 2));
  v72 = v71;
  if ( v103 )
    sub_B91220((__int64)&v103, v103);
  if ( v83 != v81 )
  {
    v73 = v81;
    do
    {
      v74 = v73++;
      v75 = &v108[2 * v74];
      *v75 = v69;
      *((_DWORD *)v75 + 2) = v72;
    }
    while ( v73 != v83 );
  }
  *((_QWORD *)&v79 + 1) = (unsigned int)v109;
  *(_QWORD *)&v79 = v108;
  v76 = sub_33FC220(a1[1], 156, (__int64)&v97, v93, v94, v70, v79);
  if ( v108 != v110 )
    _libc_free((unsigned __int64)v108);
  if ( v97 )
    sub_B91220((__int64)&v97, v97);
  return v76;
}
