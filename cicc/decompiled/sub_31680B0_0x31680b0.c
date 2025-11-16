// Function: sub_31680B0
// Address: 0x31680b0
//
__int64 __fastcall sub_31680B0(__int64 a1, __int64 a2, __int64 a3, _BYTE *a4, unsigned int a5, __int64 a6)
{
  int v9; // ecx
  __int64 v10; // rsi
  int v11; // ecx
  unsigned int v12; // edx
  __int64 *v13; // rax
  __int64 v14; // r8
  __int64 v15; // r13
  int v17; // eax
  unsigned __int8 v18; // al
  __int64 v19; // rdx
  const char *v20; // rcx
  __int64 v21; // rax
  __int64 v22; // rdx
  int v23; // ecx
  __int64 v24; // rax
  unsigned int v25; // esi
  __int64 v26; // rdi
  unsigned int v27; // edx
  __int64 *v28; // rax
  __int64 v29; // rcx
  int v30; // ebx
  __int64 *v31; // r9
  int v32; // eax
  int v33; // edx
  _DWORD *v34; // rdx
  __int64 **v35; // rdi
  __int64 v36; // rax
  __int64 v37; // rbx
  void *v38; // rdx
  unsigned __int64 v39; // rdi
  __int64 v40; // rdx
  __int64 v41; // r13
  __int64 v42; // rax
  __int64 v43; // rdx
  __int64 v44; // rax
  __int64 v45; // rax
  __int64 v46; // rdx
  unsigned __int64 v47; // rax
  __int64 v48; // r15
  __int64 v49; // rbx
  __int64 v50; // rax
  __int64 v51; // rbx
  __int64 v52; // rax
  __int64 v53; // rdx
  __int64 v54; // rax
  __int64 v55; // rdx
  unsigned __int8 v56; // dl
  __int64 *v57; // rcx
  __int64 v58; // rax
  int v59; // esi
  unsigned int v60; // ebx
  __int64 v61; // r13
  __int64 v62; // rax
  char v63; // dl
  __int64 v64; // rax
  int v65; // eax
  __int64 v66; // r9
  __int64 v67; // r8
  int v68; // esi
  unsigned __int8 v69; // al
  __int64 *v70; // rdx
  unsigned __int8 v71; // al
  __int64 v72; // rdx
  __int64 v73; // rdx
  __int64 v74; // rax
  __int64 v75; // rdx
  __int64 v76; // r10
  __int64 v77; // rax
  __int64 v78; // r9
  __int64 v79; // rdx
  unsigned __int64 v80; // r8
  __int64 v81; // rdx
  __int64 *v82; // rsi
  __int64 v83; // rax
  int v84; // edi
  void *v85; // rax
  size_t v86; // rdx
  __int64 v87; // r8
  __int64 v88; // r9
  size_t v89; // rbx
  __int64 *v90; // rsi
  __int64 v91; // rax
  __int64 v92; // rax
  __int64 v93; // rbx
  char v94; // al
  __int64 *v95; // rdi
  __int64 *v96; // rax
  int v97; // [rsp+28h] [rbp-148h]
  void *src; // [rsp+30h] [rbp-140h]
  void *srca; // [rsp+30h] [rbp-140h]
  void *srcb; // [rsp+30h] [rbp-140h]
  const char *v102; // [rsp+50h] [rbp-120h]
  __int64 v103; // [rsp+50h] [rbp-120h]
  __int64 v104; // [rsp+50h] [rbp-120h]
  __int64 v105; // [rsp+58h] [rbp-118h]
  __int64 v108; // [rsp+78h] [rbp-F8h] BYREF
  __int64 *v109; // [rsp+80h] [rbp-F0h] BYREF
  __int64 v110; // [rsp+88h] [rbp-E8h]
  __int64 v111; // [rsp+90h] [rbp-E0h]
  _BYTE v112[24]; // [rsp+98h] [rbp-D8h] BYREF
  __int64 *v113; // [rsp+B0h] [rbp-C0h] BYREF
  __int64 v114; // [rsp+B8h] [rbp-B8h]
  __int64 v115; // [rsp+C0h] [rbp-B0h] BYREF
  __int64 v116; // [rsp+C8h] [rbp-A8h] BYREF
  _DWORD *v117; // [rsp+D0h] [rbp-A0h]
  __int64 v118; // [rsp+D8h] [rbp-98h]
  const void **v119; // [rsp+E0h] [rbp-90h]

  v9 = *(_DWORD *)(a6 + 24);
  v10 = *(_QWORD *)(a6 + 8);
  if ( v9 )
  {
    v11 = v9 - 1;
    v12 = v11 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v13 = (__int64 *)(v10 + 16LL * v12);
    v14 = *v13;
    if ( a2 == *v13 )
    {
LABEL_3:
      v15 = v13[1];
      if ( v15 )
        return v15;
    }
    else
    {
      v17 = 1;
      while ( v14 != -4096 )
      {
        v84 = v17 + 1;
        v12 = v11 & (v17 + v12);
        v13 = (__int64 *)(v10 + 16LL * v12);
        v14 = *v13;
        if ( a2 == *v13 )
          goto LABEL_3;
        v17 = v84;
      }
    }
  }
  v18 = *(_BYTE *)(a2 + 8);
  if ( v18 == 12 )
  {
    v118 = 0x100000000LL;
    v109 = (__int64 *)v112;
    v110 = 0;
    v113 = (__int64 *)&unk_49DD288;
    v111 = 16;
    v114 = 2;
    v115 = 0;
    v116 = 0;
    v117 = 0;
    v119 = (const void **)&v109;
    sub_CB5980((__int64)&v113, 0, 0, 0);
    v34 = v117;
    if ( (unsigned __int64)(v116 - (_QWORD)v117) <= 5 )
    {
      v35 = (__int64 **)sub_CB6200((__int64)&v113, "__int_", 6u);
    }
    else
    {
      *v117 = 1852399455;
      v35 = &v113;
      *((_WORD *)v34 + 2) = 24436;
      v117 = (_DWORD *)((char *)v117 + 6);
    }
    sub_CB59D0((__int64)v35, *(_DWORD *)(a2 + 8) >> 8);
    v36 = sub_B9B140(*(__int64 **)a2, *v119, (size_t)v119[1]);
    v37 = sub_B91420(v36);
    src = v38;
    v113 = (__int64 *)&unk_49DD388;
    sub_CB5840((__int64)&v113);
    v39 = (unsigned __int64)v109;
    v40 = (__int64)src;
    if ( v109 != (__int64 *)v112 )
      goto LABEL_36;
    goto LABEL_37;
  }
  if ( v18 <= 3u )
  {
    v19 = 8;
    v20 = "__float_";
    if ( v18 == 2 )
    {
LABEL_10:
      v102 = v20;
      v105 = v19;
LABEL_20:
      v21 = sub_9208B0(a3, a2);
      v114 = v22;
      v113 = (__int64 *)v21;
      v23 = sub_CA1930(&v113);
      v24 = sub_ADC9A0(a1, (__int64)v102, v105, v23, 4, 64, 0);
LABEL_21:
      v15 = v24;
      goto LABEL_22;
    }
    if ( v18 == 3 )
    {
      v19 = 9;
      v20 = "__double_";
      goto LABEL_10;
    }
    goto LABEL_12;
  }
  if ( v18 == 5 || (v18 & 0xFD) == 4 )
  {
LABEL_12:
    v105 = 16;
    v102 = "__floating_type_";
    goto LABEL_13;
  }
  if ( v18 == 14 )
  {
    v105 = 11;
    v102 = "PointerType";
LABEL_40:
    BYTE4(v109) = 0;
    v41 = 8LL << sub_AE5020(a3, a2);
    v42 = sub_9208B0(a3, a2);
    v114 = v43;
    v113 = (__int64 *)v42;
    v44 = sub_CA1930(&v113);
    v15 = sub_ADCA40(a1, 0, v44, v41, (__int64)v109, 0, (__int64)v102, v105);
    goto LABEL_22;
  }
  v105 = 11;
  v102 = "UnknownType";
  if ( v18 == 15 )
  {
    if ( !*(_QWORD *)(a2 + 24) )
    {
      v105 = 20;
      v102 = "__LiteralStructType_";
      goto LABEL_19;
    }
    v85 = (void *)sub_BCB490(a2);
    v113 = &v116;
    v114 = 0;
    v89 = v86;
    v115 = 16;
    if ( v86 > 0x10 )
    {
      srcb = v85;
      sub_C8D290((__int64)&v113, &v116, v86, 1u, v87, v88);
      v85 = srcb;
      v95 = (__int64 *)((char *)v113 + v114);
    }
    else
    {
      v90 = &v116;
      if ( !v86 )
        goto LABEL_77;
      v95 = &v116;
    }
    memcpy(v95, v85, v89);
    v96 = v113;
    v114 += v89;
    v86 = v114;
    v90 = (__int64 *)((char *)v113 + v114);
    if ( v113 != (__int64 *)((char *)v113 + v114) )
    {
      do
      {
        if ( *(_BYTE *)v96 == 58 || *(_BYTE *)v96 == 46 )
          *(_BYTE *)v96 = 95;
        v96 = (__int64 *)((char *)v96 + 1);
      }
      while ( v90 != v96 );
      v86 = v114;
      v90 = v113;
    }
LABEL_77:
    v91 = sub_B9B140(*(__int64 **)a2, v90, v86);
    v92 = sub_B91420(v91);
    v39 = (unsigned __int64)v113;
    v37 = v92;
    if ( v113 != &v116 )
    {
      src = (void *)v40;
LABEL_36:
      _libc_free(v39);
      v40 = (__int64)src;
    }
LABEL_37:
    v18 = *(_BYTE *)(a2 + 8);
    v102 = (const char *)v37;
    v105 = v40;
    if ( v18 == 12 )
    {
      v24 = sub_ADC9A0(a1, v37, v40, *(_DWORD *)(a2 + 8) >> 8, 5, 64, 0);
      goto LABEL_21;
    }
LABEL_13:
    if ( v18 <= 3u || v18 == 5 )
      goto LABEL_20;
  }
LABEL_19:
  if ( (v18 & 0xFD) == 4 )
    goto LABEL_20;
  if ( v18 == 14 )
    goto LABEL_40;
  if ( v18 == 15 )
  {
    v51 = 8LL << sub_AE5260(a3, a2);
    v52 = sub_9208B0(a3, a2);
    v114 = v53;
    v113 = (__int64 *)v52;
    v54 = sub_CA1930(&v113);
    LODWORD(v55) = (_DWORD)a4;
    if ( *a4 != 16 )
    {
      v56 = *(a4 - 16);
      if ( (v56 & 2) != 0 )
        v57 = (__int64 *)*((_QWORD *)a4 - 4);
      else
        v57 = (__int64 *)&a4[-8 * ((v56 >> 2) & 0xF) - 16];
      v55 = *v57;
    }
    v58 = sub_ADE0A0(a1, a4, (__int64)v102, v105, v55, a5, v54, v51, 64, 0, 0, 0, 0, (__int64)byte_3F871B3, 0);
    v59 = *(_DWORD *)(a2 + 12);
    v108 = v58;
    v113 = &v115;
    v114 = 0x1000000000LL;
    if ( v59 )
    {
      v60 = 0;
      do
      {
        v61 = sub_31680B0(a1, *(_QWORD *)(*(_QWORD *)(a2 + 16) + 8LL * v60), a3, a4, a5, a6);
        v62 = 16LL * v60 + sub_AE4AC0(a3, a2) + 24;
        v63 = *(_BYTE *)(v62 + 8);
        v64 = 8LL * *(_QWORD *)v62;
        LOBYTE(v110) = v63;
        v109 = (__int64 *)v64;
        v103 = sub_CA1930(&v109);
        v65 = sub_AF18D0(v61);
        v66 = *(_QWORD *)(v61 + 24);
        LODWORD(v67) = (_DWORD)a4;
        v68 = v65;
        if ( *a4 != 16 )
        {
          v69 = *(a4 - 16);
          if ( (v69 & 2) != 0 )
            v70 = (__int64 *)*((_QWORD *)a4 - 4);
          else
            v70 = (__int64 *)&a4[-8 * ((v69 >> 2) & 0xF) - 16];
          v67 = *v70;
        }
        v71 = *(_BYTE *)(v61 - 16);
        if ( (v71 & 2) != 0 )
          v72 = *(_QWORD *)(v61 - 32);
        else
          v72 = v61 - 16 - 8LL * ((v71 >> 2) & 0xF);
        v73 = *(_QWORD *)(v72 + 16);
        if ( v73 )
        {
          v97 = v67;
          srca = *(void **)(v61 + 24);
          v74 = sub_B91420(v73);
          v66 = (__int64)srca;
          LODWORD(v67) = v97;
          v76 = v75;
          v73 = v74;
        }
        else
        {
          v76 = 0;
        }
        v77 = sub_ADCBB0(a1, a4, v73, v76, v67, a5, v66, v68, v103, 64, v61);
        v79 = (unsigned int)v114;
        v80 = (unsigned int)v114 + 1LL;
        if ( v80 > HIDWORD(v114) )
        {
          v104 = v77;
          sub_C8D5F0((__int64)&v113, &v115, (unsigned int)v114 + 1LL, 8u, v80, v78);
          v79 = (unsigned int)v114;
          v77 = v104;
        }
        ++v60;
        v113[v79] = v77;
        v81 = (unsigned int)(v114 + 1);
        LODWORD(v114) = v114 + 1;
      }
      while ( v60 < *(_DWORD *)(a2 + 12) );
      v82 = v113;
    }
    else
    {
      v82 = &v115;
      v81 = 0;
    }
    v83 = sub_ADCD70(a1, (__int64)v82, v81);
    sub_ADEAE0(a1, &v108, v83, 0);
    v15 = v108;
    if ( v113 != &v115 )
      _libc_free((unsigned __int64)v113);
  }
  else
  {
    v45 = sub_9208B0(a3, a2);
    v114 = v46;
    v113 = (__int64 *)v45;
    v15 = sub_ADC9A0(a1, (__int64)v102, v105, 8, 8, 64, 0);
    if ( (unsigned __int64)sub_CA1930(&v113) > 8 )
    {
      if ( (sub_CA1930(&v113) & 7) != 0 )
      {
        v93 = sub_CA1930(&v113);
        v94 = sub_CA1930(&v113);
        LOBYTE(v114) = 0;
        v113 = (__int64 *)(v93 + 8 - (v94 & 7));
      }
      v47 = sub_CA1930(&v113);
      v109 = (__int64 *)sub_ADD550(a1, 0, v47 >> 3);
      v48 = sub_ADCD70(a1, (__int64)&v109, 1);
      v49 = 1LL << sub_AE5260(a3, a2);
      v50 = sub_CA1930(&v113);
      v15 = sub_ADE2A0(a1, v50, v49, v15, v48);
    }
  }
LABEL_22:
  v25 = *(_DWORD *)(a6 + 24);
  v113 = (__int64 *)a2;
  v114 = v15;
  if ( !v25 )
  {
    ++*(_QWORD *)a6;
    v109 = 0;
LABEL_82:
    v25 *= 2;
    goto LABEL_83;
  }
  v26 = *(_QWORD *)(a6 + 8);
  v27 = (v25 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v28 = (__int64 *)(v26 + 16LL * v27);
  v29 = *v28;
  if ( a2 == *v28 )
    return v15;
  v30 = 1;
  v31 = 0;
  while ( v29 != -4096 )
  {
    if ( v29 != -8192 || v31 )
      v28 = v31;
    v27 = (v25 - 1) & (v30 + v27);
    v29 = *(_QWORD *)(v26 + 16LL * v27);
    if ( a2 == v29 )
      return v15;
    ++v30;
    v31 = v28;
    v28 = (__int64 *)(v26 + 16LL * v27);
  }
  if ( !v31 )
    v31 = v28;
  v32 = *(_DWORD *)(a6 + 16);
  ++*(_QWORD *)a6;
  v33 = v32 + 1;
  v109 = v31;
  if ( 4 * (v32 + 1) >= 3 * v25 )
    goto LABEL_82;
  if ( v25 - *(_DWORD *)(a6 + 20) - v33 <= v25 >> 3 )
  {
LABEL_83:
    sub_3167ED0(a6, v25);
    sub_31633D0(a6, (__int64 *)&v113, &v109);
    a2 = (__int64)v113;
    v31 = v109;
    v33 = *(_DWORD *)(a6 + 16) + 1;
  }
  *(_DWORD *)(a6 + 16) = v33;
  if ( *v31 != -4096 )
    --*(_DWORD *)(a6 + 20);
  *v31 = a2;
  v31[1] = v114;
  return v15;
}
