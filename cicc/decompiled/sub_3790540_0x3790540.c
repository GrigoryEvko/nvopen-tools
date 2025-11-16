// Function: sub_3790540
// Address: 0x3790540
//
unsigned __int8 *__fastcall sub_3790540(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7)
{
  int v7; // r15d
  __int64 v8; // rbx
  __int64 v9; // rsi
  int v10; // r13d
  unsigned __int16 *v11; // rdx
  int v12; // eax
  __int64 v13; // rdx
  __int64 v14; // rcx
  unsigned __int8 *v15; // r12
  unsigned __int64 v17; // r14
  unsigned int v18; // r12d
  _BYTE *v19; // rax
  __int64 v20; // rsi
  _BYTE *v21; // rdx
  _BYTE *j; // r13
  unsigned int v23; // r13d
  __int64 v24; // r13
  unsigned __int64 v25; // r15
  __int128 v26; // rax
  unsigned __int8 *v27; // rax
  int v28; // edx
  int v29; // edi
  unsigned __int8 *v30; // rdx
  __int64 v31; // rax
  _BYTE *v32; // rax
  unsigned int v33; // ebx
  __int64 v34; // r9
  _QWORD *v35; // r14
  int v36; // edx
  int v37; // r15d
  __int64 v38; // rax
  _BYTE *v39; // rcx
  __int128 v40; // rax
  unsigned __int64 v41; // rdi
  unsigned __int64 v42; // rax
  __int64 v43; // rax
  unsigned int v44; // eax
  int v45; // r14d
  unsigned __int64 v46; // r12
  _BYTE *v47; // rax
  _BYTE *v48; // rcx
  _BYTE *i; // rdx
  __int64 v50; // rdi
  int v51; // edx
  __int64 v52; // rsi
  __int64 v53; // r9
  int v54; // edi
  __int64 v55; // rax
  unsigned int v56; // edx
  __int64 v57; // rax
  _BYTE *v58; // rax
  unsigned __int8 *v59; // rax
  _QWORD *v60; // r12
  __int128 v61; // rax
  __int64 v62; // r9
  int v63; // kr00_4
  __int64 v64; // rdx
  unsigned __int8 *v65; // rax
  __int64 v66; // r9
  __int64 v67; // rbx
  int v68; // edx
  int v69; // r14d
  __int64 v70; // rdx
  unsigned __int64 v71; // r8
  int v72; // ecx
  _BYTE *v73; // rdx
  __int64 v74; // rdi
  unsigned __int8 *v75; // r14
  int v76; // edx
  int v77; // r13d
  __int64 v78; // r9
  __int64 v79; // rdx
  unsigned int v80; // eax
  __int64 v81; // rbx
  int v82; // ecx
  unsigned __int64 v83; // r8
  __int64 v84; // rsi
  _BYTE *v85; // rdx
  _QWORD *v86; // r14
  __int128 v87; // rax
  __int64 v88; // r9
  __int128 v89; // [rsp-20h] [rbp-340h]
  __int128 v90; // [rsp-10h] [rbp-330h]
  __int128 v91; // [rsp-10h] [rbp-330h]
  __int128 v92; // [rsp-10h] [rbp-330h]
  __int64 v93; // [rsp-8h] [rbp-328h]
  char v94; // [rsp+18h] [rbp-308h]
  unsigned int v95; // [rsp+1Ch] [rbp-304h]
  __int64 v96; // [rsp+20h] [rbp-300h]
  __int64 v97; // [rsp+30h] [rbp-2F0h]
  unsigned __int8 *v98; // [rsp+30h] [rbp-2F0h]
  __int64 v99; // [rsp+38h] [rbp-2E8h]
  __int16 v100; // [rsp+40h] [rbp-2E0h]
  unsigned int v101; // [rsp+40h] [rbp-2E0h]
  __int128 v103; // [rsp+50h] [rbp-2D0h]
  _QWORD *v104; // [rsp+50h] [rbp-2D0h]
  __int128 v105; // [rsp+50h] [rbp-2D0h]
  __int64 v106; // [rsp+90h] [rbp-290h] BYREF
  __int64 v107; // [rsp+98h] [rbp-288h]
  unsigned int v108; // [rsp+B0h] [rbp-270h] BYREF
  __int64 v109; // [rsp+B8h] [rbp-268h]
  __int64 v110; // [rsp+C0h] [rbp-260h] BYREF
  int v111; // [rsp+C8h] [rbp-258h]
  _BYTE *v112; // [rsp+D0h] [rbp-250h] BYREF
  __int64 v113; // [rsp+D8h] [rbp-248h]
  _BYTE v114[256]; // [rsp+E0h] [rbp-240h] BYREF
  _BYTE *v115; // [rsp+1E0h] [rbp-140h] BYREF
  __int64 v116; // [rsp+1E8h] [rbp-138h]
  _BYTE v117[304]; // [rsp+1F0h] [rbp-130h] BYREF

  *(_QWORD *)&v103 = a2;
  *((_QWORD *)&v103 + 1) = a3;
  v8 = a2;
  v106 = a4;
  v107 = a5;
  v9 = *(_QWORD *)(a2 + 80);
  v97 = (unsigned int)a3;
  v10 = a3;
  v11 = (unsigned __int16 *)(*(_QWORD *)(v8 + 48) + 16LL * (unsigned int)a3);
  v12 = *v11;
  v13 = *((_QWORD *)v11 + 1);
  v94 = a6;
  v110 = v9;
  LOWORD(v108) = v12;
  v109 = v13;
  if ( v9 )
  {
    sub_B96E90((__int64)&v110, v9, 1);
    v12 = (unsigned __int16)v108;
  }
  v14 = (unsigned __int16)v106;
  v111 = *(_DWORD *)(v8 + 72);
  if ( (_WORD)v106 == (_WORD)v12 )
  {
    if ( (_WORD)v12 || v109 == v107 )
    {
      v15 = (unsigned __int8 *)v103;
      goto LABEL_6;
    }
  }
  else if ( (_WORD)v12 )
  {
    LOBYTE(v17) = (unsigned __int16)(v12 - 176) <= 0x34u;
    v18 = word_4456340[v12 - 1];
    if ( !(_WORD)v106 )
      goto LABEL_39;
    goto LABEL_11;
  }
  v42 = sub_3007240((__int64)&v108);
  v14 = (unsigned __int16)v106;
  v18 = v42;
  v17 = HIDWORD(v42);
  if ( !(_WORD)v106 )
  {
LABEL_39:
    v101 = v14;
    v43 = sub_3007240((__int64)&v106);
    v14 = v101;
    v95 = v43;
    if ( BYTE4(v43) != (_BYTE)v17 )
      goto LABEL_12;
    goto LABEL_40;
  }
LABEL_11:
  v95 = word_4456340[(unsigned __int16)v14 - 1];
  if ( (unsigned __int16)(v14 - 176) <= 0x34u != (_BYTE)v17 )
    goto LABEL_12;
LABEL_40:
  v44 = v95 / v18;
  v45 = v95 / v18;
  if ( !(v95 % v18) )
  {
    v46 = v44;
    v115 = v117;
    v116 = 0x1000000000LL;
    if ( v44 )
    {
      v47 = v117;
      v48 = v117;
      if ( v46 > 0x10 )
      {
        sub_C8D5F0((__int64)&v115, v117, v46, 0x10u, a5, a6);
        v48 = v115;
        v47 = &v115[16 * (unsigned int)v116];
      }
      for ( i = &v48[16 * v46]; i != v47; v47 += 16 )
      {
        if ( v47 )
        {
          *(_QWORD *)v47 = 0;
          *((_DWORD *)v47 + 2) = 0;
        }
      }
      LODWORD(v116) = v45;
    }
    v50 = *(_QWORD *)(a1 + 8);
    if ( v94 )
      v52 = (__int64)sub_3400BD0(v50, 0, (__int64)&v110, v108, v109, 0, a7, 0);
    else
      v52 = sub_3288990(v50, v108, v109);
    v54 = v51;
    v55 = (__int64)v115;
    v56 = 1;
    *(_QWORD *)v115 = v8;
    *(_DWORD *)(v55 + 8) = v10;
    if ( v45 != 1 )
    {
      do
      {
        v57 = v56++;
        v58 = &v115[16 * v57];
        *(_QWORD *)v58 = v52;
        *((_DWORD *)v58 + 2) = v54;
      }
      while ( v56 != v45 );
    }
    *((_QWORD *)&v91 + 1) = (unsigned int)v116;
    *(_QWORD *)&v91 = v115;
    v59 = sub_33FC220(*(_QWORD **)(a1 + 8), 159, (__int64)&v110, (unsigned int)v106, v107, v53, v91);
    v41 = (unsigned __int64)v115;
    v15 = v59;
    if ( v115 == v117 )
      goto LABEL_6;
LABEL_36:
    _libc_free(v41);
    goto LABEL_6;
  }
  if ( !(v18 % v95) )
  {
    v60 = *(_QWORD **)(a1 + 8);
    *(_QWORD *)&v61 = sub_3400EE0((__int64)v60, 0, (__int64)&v110, 0, a7);
    v15 = sub_3406EB0(v60, 0xA1u, (__int64)&v110, (unsigned int)v106, v107, v62, v103, v61);
    goto LABEL_6;
  }
LABEL_12:
  v19 = v114;
  v20 = 0x1000000000LL;
  v21 = v114;
  v112 = v114;
  v113 = 0x1000000000LL;
  if ( v95 )
  {
    if ( v95 > 0x10uLL )
    {
      v20 = (__int64)v114;
      sub_C8D5F0((__int64)&v112, v114, v95, 0x10u, a5, a6);
      v21 = v112;
      v19 = &v112[16 * (unsigned int)v113];
    }
    for ( j = &v21[16 * v95]; j != v19; v19 += 16 )
    {
      if ( v19 )
      {
        *(_QWORD *)v19 = 0;
        *((_DWORD *)v19 + 2) = 0;
      }
    }
    v14 = (unsigned __int16)v106;
    LODWORD(v113) = v95;
  }
  if ( (_WORD)v14 )
  {
    v99 = 0;
    v100 = word_4456580[(unsigned __int16)v14 - 1];
  }
  else
  {
    v63 = sub_3009970((__int64)&v106, v20, (__int64)v21, v14, a5);
    HIWORD(v7) = HIWORD(v63);
    v100 = v63;
    v99 = v64;
  }
  HIWORD(v23) = HIWORD(v7);
  if ( v18 > v95 )
    v18 = v95;
  if ( v18 )
  {
    v96 = v8;
    WORD1(v8) = HIWORD(v7);
    v24 = 0;
    v25 = *((_QWORD *)&v103 + 1);
    do
    {
      v104 = *(_QWORD **)(a1 + 8);
      *(_QWORD *)&v26 = sub_3400EE0((__int64)v104, v24, (__int64)&v110, 0, a7);
      LOWORD(v8) = v100;
      v25 = v97 | v25 & 0xFFFFFFFF00000000LL;
      *((_QWORD *)&v89 + 1) = v25;
      *(_QWORD *)&v89 = v96;
      v27 = sub_3406EB0(v104, 0x9Eu, (__int64)&v110, (unsigned int)v8, v99, *((__int64 *)&v26 + 1), v89, v26);
      v29 = v28;
      v30 = v27;
      v31 = v24++;
      v32 = &v112[16 * v31];
      *(_QWORD *)v32 = v30;
      *((_DWORD *)v32 + 2) = v29;
    }
    while ( v18 != v24 );
    HIWORD(v23) = WORD1(v8);
    v33 = v18;
  }
  else
  {
    v33 = 0;
  }
  LOWORD(v23) = v100;
  v115 = 0;
  LODWORD(v116) = 0;
  v35 = sub_33F17F0(*(_QWORD **)(a1 + 8), 51, (__int64)&v115, v23, v99);
  v37 = v36;
  if ( v115 )
    sub_B91220((__int64)&v115, (__int64)v115);
  if ( v33 < v95 )
  {
    v38 = 16LL * v33;
    do
    {
      v39 = v112;
      *(_QWORD *)&v112[v38] = v35;
      *(_DWORD *)&v39[v38 + 8] = v37;
      v38 += 16;
    }
    while ( v38 != 16 * (v33 + (unsigned __int64)(v95 - 1 - v33) + 1) );
  }
  *((_QWORD *)&v90 + 1) = (unsigned int)v113;
  *(_QWORD *)&v90 = v112;
  *(_QWORD *)&v40 = sub_33FC220(*(_QWORD **)(a1 + 8), 156, (__int64)&v110, v106, v107, v34, v90);
  v105 = v40;
  if ( v94 )
  {
    v116 = 0x1000000000LL;
    v115 = v117;
    v65 = sub_34015B0(*(_QWORD *)(a1 + 8), (__int64)&v110, v23, v99, 0, 0, a7);
    v67 = v18;
    v69 = v68;
    v70 = (unsigned int)v116;
    v71 = v18 + (unsigned __int64)(unsigned int)v116;
    v72 = v116;
    if ( v71 > HIDWORD(v116) )
    {
      v98 = v65;
      sub_C8D5F0((__int64)&v115, v117, v18 + (unsigned __int64)(unsigned int)v116, 0x10u, v71, v66);
      v70 = (unsigned int)v116;
      v65 = v98;
      v72 = v116;
    }
    v73 = &v115[16 * v70];
    if ( v18 )
    {
      do
      {
        if ( v73 )
        {
          *(_QWORD *)v73 = v65;
          *((_DWORD *)v73 + 2) = v69;
        }
        v73 += 16;
        --v67;
      }
      while ( v67 );
      v72 = v116;
    }
    v74 = *(_QWORD *)(a1 + 8);
    LODWORD(v116) = v18 + v72;
    v75 = sub_3400BD0(v74, 0, (__int64)&v110, v23, v99, 0, a7, 0);
    v77 = v76;
    v78 = v93;
    v79 = (unsigned int)v116;
    v80 = v95 - v18;
    v81 = v95 - v18;
    v82 = v116;
    v83 = v81 + (unsigned int)v116;
    if ( v83 > HIDWORD(v116) )
    {
      sub_C8D5F0((__int64)&v115, v117, v81 + (unsigned int)v116, 0x10u, v83, v93);
      v79 = (unsigned int)v116;
      v80 = v95 - v18;
      v82 = v116;
    }
    v84 = (__int64)v115;
    v85 = &v115[16 * v79];
    if ( v95 != v18 )
    {
      do
      {
        if ( v85 )
        {
          *(_QWORD *)v85 = v75;
          *((_DWORD *)v85 + 2) = v77;
        }
        v85 += 16;
        --v81;
      }
      while ( v81 );
      v84 = (__int64)v115;
      v82 = v116;
    }
    LODWORD(v116) = v82 + v80;
    v86 = *(_QWORD **)(a1 + 8);
    *((_QWORD *)&v92 + 1) = v82 + v80;
    *(_QWORD *)&v92 = v84;
    *(_QWORD *)&v87 = sub_33FC220(v86, 156, (__int64)&v110, v106, v107, v78, v92);
    v15 = sub_3406EB0(v86, 0xBAu, (__int64)&v110, (unsigned int)v106, v107, v88, v105, v87);
    if ( v115 != v117 )
      _libc_free((unsigned __int64)v115);
  }
  else
  {
    v15 = (unsigned __int8 *)v40;
  }
  v41 = (unsigned __int64)v112;
  if ( v112 != v114 )
    goto LABEL_36;
LABEL_6:
  if ( v110 )
    sub_B91220((__int64)&v110, v110);
  return v15;
}
