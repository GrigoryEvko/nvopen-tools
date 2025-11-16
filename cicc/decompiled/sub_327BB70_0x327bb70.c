// Function: sub_327BB70
// Address: 0x327bb70
//
__int64 __fastcall sub_327BB70(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  __int64 v4; // r12
  __int16 *v6; // rdx
  __int16 v7; // ax
  __int64 v8; // rdx
  unsigned __int16 v9; // ax
  __int64 v10; // rdx
  unsigned int v11; // r15d
  __int64 v12; // rdx
  __int64 v13; // r13
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // rdx
  _BYTE *v18; // r14
  __int64 v19; // r13
  __int16 *v21; // rdx
  __int16 v22; // ax
  __int64 v23; // rdx
  unsigned __int64 v24; // r14
  void *v25; // r8
  __int64 v26; // r9
  unsigned __int64 *v27; // r14
  unsigned int *v28; // rdx
  char *v29; // rsi
  unsigned int v30; // r8d
  unsigned int v31; // ecx
  __int64 v32; // rax
  __int64 v33; // r9
  __int64 v34; // r8
  __int64 v35; // rax
  _QWORD *v36; // r10
  unsigned int *v37; // rdx
  char v38; // r9
  char *v39; // rdi
  __int64 v40; // rax
  unsigned int v41; // eax
  unsigned int v42; // ecx
  unsigned __int64 *v43; // r13
  unsigned __int64 *v44; // r15
  __int64 v45; // r14
  _QWORD *v46; // r12
  __int64 *v47; // r15
  unsigned int v48; // r13d
  unsigned __int16 v49; // ax
  __int64 v50; // rdx
  unsigned int v51; // r13d
  unsigned __int16 v52; // ax
  int v53; // r13d
  char v54; // dl
  __int16 *v55; // r14
  unsigned int v56; // r15d
  __int64 *v57; // rax
  int v58; // esi
  int *v59; // rax
  unsigned __int64 v60; // r9
  int v61; // edx
  int v62; // edi
  char *v63; // rdi
  int v64; // kr00_4
  __int64 v65; // rdx
  int v66; // r9d
  __int64 v67; // r13
  __int64 v68; // rdx
  __int64 v69; // r14
  __int64 v70; // rax
  __int64 v71; // rdx
  __int128 v72; // [rsp-10h] [rbp-1F0h]
  __int64 v73; // [rsp+0h] [rbp-1E0h]
  __int64 v74; // [rsp+8h] [rbp-1D8h]
  unsigned int v75; // [rsp+18h] [rbp-1C8h]
  int v76; // [rsp+20h] [rbp-1C0h]
  __int64 v77; // [rsp+28h] [rbp-1B8h]
  __int64 v78; // [rsp+28h] [rbp-1B8h]
  _QWORD *src; // [rsp+30h] [rbp-1B0h]
  unsigned __int16 srca; // [rsp+30h] [rbp-1B0h]
  void *srcb; // [rsp+30h] [rbp-1B0h]
  unsigned int n; // [rsp+48h] [rbp-198h]
  __int16 v85; // [rsp+6Eh] [rbp-172h] BYREF
  unsigned int v86; // [rsp+70h] [rbp-170h] BYREF
  __int64 v87; // [rsp+78h] [rbp-168h]
  __int64 v88; // [rsp+80h] [rbp-160h]
  __int64 v89; // [rsp+88h] [rbp-158h]
  _DWORD v90[4]; // [rsp+90h] [rbp-150h] BYREF
  char v91; // [rsp+A0h] [rbp-140h]
  unsigned __int64 v92; // [rsp+B0h] [rbp-130h] BYREF
  _DWORD v93[2]; // [rsp+B8h] [rbp-128h]
  unsigned __int64 v94; // [rsp+C0h] [rbp-120h]
  unsigned int v95; // [rsp+C8h] [rbp-118h]
  unsigned __int64 v96; // [rsp+D0h] [rbp-110h] BYREF
  _DWORD v97[2]; // [rsp+D8h] [rbp-108h]
  unsigned __int64 v98; // [rsp+E0h] [rbp-100h]
  unsigned int v99; // [rsp+E8h] [rbp-F8h]
  __int64 v100; // [rsp+F0h] [rbp-F0h] BYREF
  int **v101; // [rsp+F8h] [rbp-E8h]
  __int64 (__fastcall *v102)(const __m128i **, const __m128i *, int); // [rsp+100h] [rbp-E0h]
  __int64 (__fastcall *v103)(__int64, unsigned int *); // [rsp+108h] [rbp-D8h]
  void *v104; // [rsp+110h] [rbp-D0h] BYREF
  __int64 v105; // [rsp+118h] [rbp-C8h]
  _BYTE v106[64]; // [rsp+120h] [rbp-C0h] BYREF
  int *v107; // [rsp+160h] [rbp-80h] BYREF
  __int64 v108; // [rsp+168h] [rbp-78h]
  _BYTE v109[112]; // [rsp+170h] [rbp-70h] BYREF

  v4 = a2;
  v6 = *(__int16 **)(a1 + 48);
  v7 = *v6;
  v8 = *((_QWORD *)v6 + 1);
  LOWORD(v86) = v7;
  v87 = v8;
  if ( !v7 )
  {
    if ( !sub_3007100((__int64)&v86) )
      goto LABEL_11;
    goto LABEL_22;
  }
  if ( (unsigned __int16)(v7 - 176) <= 0x34u )
  {
LABEL_22:
    sub_CA17B0(
      "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::"
      "getVectorElementCount() instead");
    v9 = v86;
    if ( (_WORD)v86 )
    {
      if ( (unsigned __int16)(v86 - 176) > 0x34u )
      {
        v10 = (unsigned __int16)v86 - 1;
        v11 = word_4456340[v10];
LABEL_4:
        if ( (unsigned __int16)(v9 - 17) <= 0xD3u )
        {
          v9 = word_4456580[v10];
          v12 = 0;
        }
        else
        {
          v12 = v87;
        }
        goto LABEL_6;
      }
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
      goto LABEL_3;
    }
LABEL_11:
    v11 = sub_3007130((__int64)&v86, a2);
    goto LABEL_12;
  }
LABEL_3:
  v9 = v86;
  v10 = (unsigned __int16)v86 - 1;
  v11 = word_4456340[v10];
  if ( (_WORD)v86 )
    goto LABEL_4;
LABEL_12:
  if ( !sub_30070B0((__int64)&v86) )
  {
    LOWORD(v107) = 0;
    v108 = v87;
    goto LABEL_14;
  }
  v9 = sub_3009970((__int64)&v86, a2, v14, v15, v16);
LABEL_6:
  LOWORD(v107) = v9;
  v108 = v12;
  if ( !v9 )
  {
LABEL_14:
    v88 = sub_3007260((__int64)&v107);
    LODWORD(v13) = v88;
    v89 = v17;
    goto LABEL_15;
  }
  if ( v9 == 1 || (unsigned __int16)(v9 - 504) <= 7u )
    BUG();
  v13 = *(_QWORD *)&byte_444C4A0[16 * v9 - 16];
LABEL_15:
  v18 = (_BYTE *)sub_2E79000(*(__int64 **)(a2 + 40));
  if ( (_WORD)v86 )
  {
    if ( (unsigned __int16)(v86 - 2) > 7u
      && (unsigned __int16)(v86 - 17) > 0x6Cu
      && (unsigned __int16)(v86 - 176) > 0x1Fu )
    {
      return 0;
    }
  }
  else if ( !sub_3007070((__int64)&v86) )
  {
    return 0;
  }
  if ( *v18 )
    return 0;
  v21 = *(__int16 **)(a1 + 48);
  v22 = *v21;
  v23 = *((_QWORD *)v21 + 1);
  LOWORD(v107) = v22;
  v108 = v23;
  if ( v22 )
  {
    if ( (unsigned __int16)(v22 - 176) > 0x34u )
    {
LABEL_28:
      v24 = word_4456340[(unsigned __int16)v107 - 1];
      goto LABEL_29;
    }
  }
  else if ( !sub_3007100((__int64)&v107) )
  {
    goto LABEL_78;
  }
  sub_CA17B0(
    "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::ge"
    "tVectorElementCount() instead");
  if ( (_WORD)v107 )
  {
    if ( (unsigned __int16)((_WORD)v107 - 176) <= 0x34u )
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
    goto LABEL_28;
  }
LABEL_78:
  v24 = (unsigned int)sub_3007130((__int64)&v107, a2);
LABEL_29:
  v25 = *(void **)(a1 + 96);
  v26 = 4 * v24;
  v104 = v106;
  v105 = 0x1000000000LL;
  if ( v24 > 0x10 )
  {
    srcb = v25;
    sub_C8D5F0((__int64)&v104, v106, v24, 4u, (__int64)v25, v26);
    v26 = 4 * v24;
    v25 = srcb;
    v63 = (char *)v104 + 4 * (unsigned int)v105;
LABEL_117:
    memcpy(v63, v25, v26);
    LODWORD(v26) = v105;
    goto LABEL_31;
  }
  if ( v26 )
  {
    v63 = v106;
    goto LABEL_117;
  }
LABEL_31:
  v93[0] = 1;
  LODWORD(v105) = v26 + v24;
  v27 = &v92;
  v92 = 0;
  v95 = 1;
  v94 = 0;
  do
  {
    LODWORD(v108) = v11;
    if ( v11 > 0x40 )
      sub_C43690((__int64)&v107, 0, 0);
    else
      v107 = 0;
    if ( *((_DWORD *)v27 + 2) > 0x40u && *v27 )
      j_j___libc_free_0_0(*v27);
    v27 += 2;
    *(v27 - 2) = (unsigned __int64)v107;
    *((_DWORD *)v27 - 2) = v108;
  }
  while ( &v96 != v27 );
  v28 = (unsigned int *)v104;
  v29 = (char *)v104 + 4 * (unsigned int)v105;
  if ( v104 != v29 )
  {
    while ( 1 )
    {
      v30 = *v28;
      if ( (*v28 & 0x80000000) != 0 )
        goto LABEL_41;
      v31 = v30 - v11;
      if ( v11 > v30 )
        v31 = *v28;
      v32 = 4LL * (v11 <= v30);
      v33 = 1LL << v31;
      v34 = *(_QWORD *)&v93[v32 - 2];
      if ( v93[v32] <= 0x40u )
      {
        *(_QWORD *)&v93[v32 - 2] = v33 | v34;
LABEL_41:
        if ( v29 == (char *)++v28 )
          break;
      }
      else
      {
        ++v28;
        *(_QWORD *)(v34 + 8LL * (v31 >> 6)) |= v33;
        if ( v29 == (char *)v28 )
          break;
      }
    }
  }
  v35 = *(unsigned int *)(a1 + 64);
  v36 = *(_QWORD **)(a1 + 40);
  v97[0] = 1;
  v96 = 0;
  v99 = 1;
  v98 = 0;
  if ( &v36[5 * v35] != v36 )
  {
    src = &v36[5 * v35];
    v76 = v13;
    v43 = &v92;
    v75 = v11;
    v44 = &v96;
    v45 = v4;
    v46 = v36;
    do
    {
      sub_33DC2B0(&v107, v45, *v46, v46[1], v43, 0);
      if ( *((_DWORD *)v44 + 2) > 0x40u && *v44 )
        j_j___libc_free_0_0(*v44);
      v46 += 5;
      v43 += 2;
      v44 += 2;
      *(v44 - 2) = (unsigned __int64)v107;
      *((_DWORD *)v44 - 2) = v108;
    }
    while ( src != v46 && v43 != &v96 && v44 != (unsigned __int64 *)&v100 );
    LODWORD(v13) = v76;
    v11 = v75;
    v4 = v45;
  }
  v37 = (unsigned int *)v104;
  v38 = 0;
  v39 = (char *)v104 + 4 * (unsigned int)v105;
  if ( v104 != v39 )
  {
    do
    {
      v41 = *v37;
      if ( (*v37 & 0x80000000) == 0 )
      {
        v42 = v41 - v11;
        if ( v11 > v41 )
          v42 = *v37;
        if ( v97[4 * (v11 <= v41)] <= 0x40u )
          v40 = *(_QWORD *)&v97[4 * (v11 <= v41) - 2];
        else
          v40 = *(_QWORD *)(*(_QWORD *)((char *)&v97[-2] + (v41 >= v11 ? 0x10 : 0)) + 8LL * (v42 >> 6));
        if ( (v40 & (1LL << v42)) != 0 )
        {
          *v37 = -2;
          v38 = 1;
        }
      }
      ++v37;
    }
    while ( v39 != (char *)v37 );
    if ( v38 )
    {
      v107 = (int *)v109;
      v108 = 0x1000000000LL;
      sub_9B87B0((char *)v104, (unsigned int)v105, (__int64)&v107);
      v47 = *(__int64 **)(v4 + 64);
      n = v108;
      v48 = (unsigned int)v105 / (unsigned int)v108 * v13;
      switch ( v48 )
      {
        case 1u:
          v49 = 2;
          v50 = 0;
          break;
        case 2u:
          v49 = 3;
          v50 = 0;
          break;
        case 4u:
          v49 = 4;
          v50 = 0;
          break;
        case 8u:
          v49 = 5;
          v50 = 0;
          break;
        case 0x10u:
          v49 = 6;
          v50 = 0;
          break;
        case 0x20u:
          v49 = 7;
          v50 = 0;
          break;
        case 0x40u:
          v49 = 8;
          v50 = 0;
          break;
        case 0x80u:
          v49 = 9;
          v50 = 0;
          break;
        default:
          v49 = sub_3007020(v47, v48);
          v47 = *(__int64 **)(v4 + 64);
          break;
      }
      v77 = v50;
      v51 = v49;
      v52 = sub_2D43050(v49, n);
      srca = v52;
      if ( v52 )
      {
        v78 = 0;
        v53 = v52;
      }
      else
      {
        v64 = sub_3009400(v47, v51, v77, n, 0);
        HIWORD(v53) = HIWORD(v64);
        srca = v64;
        v78 = v65;
        if ( !(_WORD)v64 )
        {
LABEL_121:
          if ( (_WORD)v86 && *(_QWORD *)(a3 + 8LL * (unsigned __int16)v86 + 112) )
          {
LABEL_113:
            v19 = 0;
LABEL_114:
            if ( v107 != (int *)v109 )
              _libc_free((unsigned __int64)v107);
            goto LABEL_62;
          }
LABEL_100:
          v54 = 0;
          v85 = 256;
          v55 = &v85;
          HIWORD(v56) = HIWORD(v53);
          while ( 1 )
          {
            v57 = *(__int64 **)(a1 + 40);
            if ( v54 )
            {
              v74 = v57[5];
              v58 = v108;
              v73 = v57[6];
              v59 = v107;
              if ( (_DWORD)v108 )
              {
                v60 = (unsigned __int64)&v107[(unsigned int)(v108 - 1) + 1];
                do
                {
                  v61 = *v59;
                  if ( *v59 >= 0 )
                  {
                    v62 = v61 - v58;
                    if ( v61 < v58 )
                      v62 = v58 + v61;
                    *v59 = v62;
                  }
                  ++v59;
                }
                while ( v59 != (int *)v60 );
              }
            }
            else
            {
              v74 = *v57;
              v73 = v57[1];
            }
            LOWORD(v56) = srca;
            LODWORD(v100) = n;
            v101 = &v107;
            v103 = sub_325F0C0;
            v102 = sub_325DD70;
            sub_327AB70((__int64)v90, 225, v56, v78, (__int64)&v100, v4, a3, a4);
            if ( v102 )
              v102((const __m128i **)&v100, (const __m128i *)&v100, 3);
            if ( v91 )
              break;
            v55 = (__int16 *)((char *)v55 + 1);
            if ( &v86 == (unsigned int *)v55 )
              goto LABEL_113;
            v54 = *(_BYTE *)v55;
          }
          v67 = sub_33FB890(v4, v56, v78, v74, v73);
          v69 = v68;
          v100 = *(_QWORD *)(a1 + 80);
          if ( v100 )
            sub_325F5D0(&v100);
          *((_QWORD *)&v72 + 1) = v69;
          *(_QWORD *)&v72 = v67;
          LODWORD(v101) = *(_DWORD *)(a1 + 72);
          v70 = sub_33FAF80(v4, 225, (unsigned int)&v100, v90[0], v90[2], v66, v72);
          v19 = sub_33FB890(v4, v86, v87, v70, v71);
          if ( v100 )
            sub_B91220((__int64)&v100, v100);
          goto LABEL_114;
        }
      }
      if ( *(_QWORD *)(a3 + 8LL * srca + 112) )
        goto LABEL_100;
      goto LABEL_121;
    }
  }
  v19 = 0;
LABEL_62:
  if ( v99 > 0x40 && v98 )
    j_j___libc_free_0_0(v98);
  if ( v97[0] > 0x40u && v96 )
    j_j___libc_free_0_0(v96);
  if ( v95 > 0x40 && v94 )
    j_j___libc_free_0_0(v94);
  if ( v93[0] > 0x40u && v92 )
    j_j___libc_free_0_0(v92);
  if ( v104 != v106 )
    _libc_free((unsigned __int64)v104);
  return v19;
}
