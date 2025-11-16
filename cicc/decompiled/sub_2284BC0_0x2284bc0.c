// Function: sub_2284BC0
// Address: 0x2284bc0
//
__int64 __fastcall sub_2284BC0(__int64 a1, __int64 *a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v10; // r12
  __int64 v11; // rbx
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // rcx
  _QWORD *v20; // rbx
  _QWORD *v21; // r15
  __int64 v22; // r14
  _QWORD *v23; // rax
  _QWORD *v24; // rdi
  _QWORD *v25; // rdi
  __int64 v26; // rdx
  __int64 v27; // rsi
  __int64 *v28; // rax
  char *v29; // r14
  __int64 v30; // rcx
  __int64 v31; // rax
  __int64 v32; // rcx
  char *v33; // rcx
  unsigned __int8 *v34; // rax
  int v35; // edx
  unsigned __int64 v36; // rdx
  __int64 v37; // rbx
  __int64 v38; // rdx
  unsigned __int8 *v39; // rax
  int v40; // edx
  unsigned __int64 v41; // rdx
  __int64 v42; // rbx
  __int64 v43; // rdx
  unsigned __int8 *v44; // rax
  int v45; // edx
  unsigned __int64 v46; // rdx
  __int64 v47; // rsi
  __int64 v48; // rdx
  unsigned __int8 *v49; // rax
  int v50; // edx
  unsigned __int64 v51; // rdx
  __int64 v52; // rbx
  __int64 v53; // rdx
  __int64 v54; // rbx
  __int64 v55; // r15
  __int64 v56; // rax
  __int64 *v57; // rax
  int v58; // edi
  int v59; // esi
  char *v60; // rax
  char *v61; // rsi
  __int64 v62; // r9
  __int64 *v63; // rdx
  _DWORD *v64; // rax
  int v65; // r8d
  __int64 v66; // rcx
  __int64 *v67; // rax
  __int64 *v68; // rdx
  __int64 v69; // rcx
  bool v70; // si
  __int64 v71; // rdi
  __int64 v72; // rcx
  unsigned int v73; // eax
  __int64 v74; // rdi
  __int64 v75; // rsi
  __int64 v76; // rdi
  char v77; // bl
  __int64 v78; // rcx
  __int64 *v79; // r14
  __int64 *v80; // rax
  __int64 *v81; // rdx
  __int64 v82; // rcx
  __int64 *v83; // r14
  char v84; // r10
  __int64 v85; // r8
  int v86; // edi
  __int64 *v87; // r9
  unsigned int v88; // esi
  __int64 *v89; // rcx
  __int64 v90; // r11
  __int64 v91; // rsi
  char v92; // r15
  __int64 v94; // rdi
  __int64 v95; // rdi
  int v96; // ecx
  __int64 v97; // rcx
  __int64 *v98; // rax
  _QWORD *v99; // rbx
  _QWORD *v100; // r12
  __int64 v101; // r13
  _QWORD *v102; // rdi
  __int64 v103; // rdx
  __int64 v104; // rsi
  void (__fastcall *v105)(_QWORD *, __int64, __int64, char *); // r8
  __int64 v109; // [rsp+30h] [rbp-170h]
  int i; // [rsp+3Ch] [rbp-164h]
  char *v113; // [rsp+58h] [rbp-148h]
  int v114; // [rsp+58h] [rbp-148h]
  __int64 v115; // [rsp+68h] [rbp-138h] BYREF
  char v116[8]; // [rsp+70h] [rbp-130h] BYREF
  __int64 v117; // [rsp+78h] [rbp-128h]
  __int64 *v118; // [rsp+80h] [rbp-120h] BYREF
  unsigned int v119; // [rsp+88h] [rbp-118h]
  _QWORD *v120; // [rsp+C0h] [rbp-E0h] BYREF
  unsigned int v121; // [rsp+C8h] [rbp-D8h]
  int v122; // [rsp+CCh] [rbp-D4h]
  __int64 *v123; // [rsp+D0h] [rbp-D0h] BYREF
  unsigned int v124; // [rsp+D8h] [rbp-C8h]
  char v125[8]; // [rsp+110h] [rbp-90h] BYREF
  unsigned __int64 v126; // [rsp+118h] [rbp-88h]
  char v127; // [rsp+12Ch] [rbp-74h]
  unsigned __int64 v128; // [rsp+148h] [rbp-58h]
  char v129; // [rsp+15Ch] [rbp-44h]

  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 16) = 2;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 64) = 2;
  *(_DWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 76) = 1;
  sub_AE6EC0(a1, (__int64)&qword_4F82400);
  v115 = *(_QWORD *)(sub_227ED20(a4, &qword_4F8A320, a3, a5) + 8);
  v109 = a6 + 136;
  sub_227BD30(a6 + 136);
  v10 = *(_QWORD *)(a6 + 408);
  v11 = v10 + 32LL * *(unsigned int *)(a6 + 416);
  while ( v10 != v11 )
  {
    while ( 1 )
    {
      v12 = *(_QWORD *)(v11 - 8);
      v11 -= 32;
      if ( v12 == -4096 || v12 == 0 || v12 == -8192 )
        break;
      sub_BD60C0((_QWORD *)(v11 + 8));
      if ( v10 == v11 )
        goto LABEL_6;
    }
  }
LABEL_6:
  *(_DWORD *)(a6 + 416) = 0;
  sub_2284740((__int64)v116, (__int64)a3, v109);
  for ( i = 0; ; ++i )
  {
    if ( !(unsigned __int8)sub_227B670(&v115, *a2, (__int64)a3) )
      continue;
    (*(void (__fastcall **)(char *, __int64, __int64 *, __int64, __int64, __int64))(*(_QWORD *)*a2 + 16LL))(
      v125,
      *a2,
      a3,
      a4,
      a5,
      a6);
    sub_227AD80(a1, (__int64)v125, v13, v14, v15, v16);
    if ( (unsigned __int8)sub_B19060(*(_QWORD *)(a6 + 8), (__int64)a3, v17, v18) )
    {
      if ( v115 )
      {
        v99 = *(_QWORD **)(v115 + 576);
        v100 = &v99[4 * *(unsigned int *)(v115 + 584)];
        if ( v99 != v100 )
        {
          v101 = *a2;
          do
          {
            v102 = v99;
            v104 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v101 + 32LL))(v101);
            v105 = *(void (__fastcall **)(_QWORD *, __int64, __int64, char *))(v99[3] & 0xFFFFFFFFFFFFFFF8LL);
            if ( (v99[3] & 2) == 0 )
              v102 = (_QWORD *)*v99;
            v99 += 4;
            v105(v102, v104, v103, v125);
          }
          while ( v100 != v99 );
        }
      }
      goto LABEL_127;
    }
    sub_227C930(a4, (__int64)a3, (__int64)v125, v19);
    if ( v115 )
    {
      v20 = *(_QWORD **)(v115 + 432);
      v21 = &v20[4 * *(unsigned int *)(v115 + 440)];
      if ( v20 != v21 )
      {
        v22 = *a2;
        do
        {
          v120 = 0;
          v23 = (_QWORD *)sub_22077B0(0x10u);
          if ( v23 )
          {
            v23[1] = a3;
            *v23 = &unk_4A08BA8;
          }
          v24 = v120;
          v120 = v23;
          if ( v24 )
            (*(void (__fastcall **)(_QWORD *))(*v24 + 8LL))(v24);
          v25 = v20;
          v27 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v22 + 32LL))(v22);
          if ( (v20[3] & 2) == 0 )
            v25 = (_QWORD *)*v20;
          (*(void (__fastcall **)(_QWORD *, __int64, __int64, _QWORD **, char *))(v20[3] & 0xFFFFFFFFFFFFFFF8LL))(
            v25,
            v27,
            v26,
            &v120,
            v125);
          if ( v120 )
            (*(void (__fastcall **)(_QWORD *))(*v120 + 8LL))(v120);
          v20 += 4;
        }
        while ( v21 != v20 );
      }
    }
    v28 = *(__int64 **)(a6 + 16);
    if ( v28 != a3 && v28 )
      goto LABEL_127;
    v29 = *(char **)(a6 + 408);
    v30 = 32LL * *(unsigned int *)(a6 + 416);
    v113 = &v29[v30];
    v31 = v30 >> 5;
    v32 = v30 >> 7;
    if ( !v32 )
      goto LABEL_94;
    v33 = &v29[128 * v32];
    do
    {
      v49 = (unsigned __int8 *)*((_QWORD *)v29 + 3);
      if ( v49 )
      {
        v50 = *v49;
        if ( (unsigned __int8)v50 > 0x1Cu )
        {
          v51 = (unsigned int)(v50 - 34);
          if ( (unsigned __int8)v51 <= 0x33u )
          {
            v52 = 0x8000000000041LL;
            if ( _bittest64(&v52, v51) )
            {
              v53 = *((_QWORD *)v49 - 4);
              if ( v53 )
              {
                if ( !*(_BYTE *)v53 && *(_QWORD *)(v53 + 24) == *((_QWORD *)v49 + 10) )
                  goto LABEL_54;
              }
            }
          }
        }
      }
      v34 = (unsigned __int8 *)*((_QWORD *)v29 + 7);
      if ( v34 )
      {
        v35 = *v34;
        if ( (unsigned __int8)v35 > 0x1Cu )
        {
          v36 = (unsigned int)(v35 - 34);
          if ( (unsigned __int8)v36 <= 0x33u )
          {
            v37 = 0x8000000000041LL;
            if ( _bittest64(&v37, v36) )
            {
              v38 = *((_QWORD *)v34 - 4);
              if ( v38 )
              {
                if ( !*(_BYTE *)v38 && *(_QWORD *)(v38 + 24) == *((_QWORD *)v34 + 10) )
                {
                  v29 += 32;
                  goto LABEL_54;
                }
              }
            }
          }
        }
      }
      v39 = (unsigned __int8 *)*((_QWORD *)v29 + 11);
      if ( v39 )
      {
        v40 = *v39;
        if ( (unsigned __int8)v40 > 0x1Cu )
        {
          v41 = (unsigned int)(v40 - 34);
          if ( (unsigned __int8)v41 <= 0x33u )
          {
            v42 = 0x8000000000041LL;
            if ( _bittest64(&v42, v41) )
            {
              v43 = *((_QWORD *)v39 - 4);
              if ( v43 )
              {
                if ( !*(_BYTE *)v43 && *(_QWORD *)(v43 + 24) == *((_QWORD *)v39 + 10) )
                {
                  v29 += 64;
                  goto LABEL_54;
                }
              }
            }
          }
        }
      }
      v44 = (unsigned __int8 *)*((_QWORD *)v29 + 15);
      if ( v44 )
      {
        v45 = *v44;
        if ( (unsigned __int8)v45 > 0x1Cu )
        {
          v46 = (unsigned int)(v45 - 34);
          if ( (unsigned __int8)v46 <= 0x33u )
          {
            v47 = 0x8000000000041LL;
            if ( _bittest64(&v47, v46) )
            {
              v48 = *((_QWORD *)v44 - 4);
              if ( v48 )
              {
                if ( !*(_BYTE *)v48 && *(_QWORD *)(v48 + 24) == *((_QWORD *)v44 + 10) )
                {
                  v29 += 96;
                  goto LABEL_54;
                }
              }
            }
          }
        }
      }
      v29 += 128;
    }
    while ( v33 != v29 );
    v31 = (v113 - v29) >> 5;
LABEL_94:
    if ( v31 == 2 )
      goto LABEL_144;
    if ( v31 == 3 )
    {
      if ( !sub_227A7C0((__int64)v29) )
      {
        v29 += 32;
LABEL_144:
        if ( !sub_227A7C0((__int64)v29) )
        {
          v29 += 32;
          goto LABEL_146;
        }
      }
LABEL_54:
      sub_227BD30(v109);
      v54 = *(_QWORD *)(a6 + 408);
      v55 = v54 + 32LL * *(unsigned int *)(a6 + 416);
      if ( v54 == v55 )
        goto LABEL_59;
      goto LABEL_55;
    }
    if ( v31 != 1 )
      goto LABEL_97;
LABEL_146:
    if ( sub_227A7C0((__int64)v29) )
      goto LABEL_54;
LABEL_97:
    sub_227BD30(v109);
    v54 = *(_QWORD *)(a6 + 408);
    v55 = v54 + 32LL * *(unsigned int *)(a6 + 416);
    if ( v55 == v54 )
    {
      *(_DWORD *)(a6 + 416) = 0;
      sub_2284740((__int64)&v120, (__int64)a3, v109);
LABEL_100:
      v77 = v121 & 1;
      if ( v121 >> 1 )
      {
        if ( v77 )
        {
          v81 = (__int64 *)v125;
          v80 = (__int64 *)&v123;
        }
        else
        {
          v78 = v124;
          v79 = v123;
          v80 = v123;
          v81 = &v123[2 * v124];
          if ( v123 == v81 )
            goto LABEL_108;
        }
        do
        {
          if ( *v80 != -8192 && *v80 != -4096 )
            break;
          v80 += 2;
        }
        while ( v80 != v81 );
      }
      else
      {
        if ( v77 )
        {
          v97 = 8;
          v98 = (__int64 *)&v123;
        }
        else
        {
          v98 = v123;
          v97 = 2LL * v124;
        }
        v80 = &v98[v97];
        v81 = v80;
      }
      if ( v77 )
      {
        v82 = 8;
        v79 = (__int64 *)&v123;
        goto LABEL_109;
      }
      v79 = v123;
      v78 = v124;
LABEL_108:
      v82 = 2 * v78;
LABEL_109:
      v83 = &v79[v82];
      if ( v83 == v80 )
      {
LABEL_124:
        v92 = v121 & 1;
        goto LABEL_125;
      }
      v84 = v117 & 1;
      while ( 1 )
      {
        v85 = *v80;
        if ( v84 )
        {
          v86 = 3;
          v87 = (__int64 *)&v118;
        }
        else
        {
          v94 = v119;
          v87 = v118;
          if ( !v119 )
            goto LABEL_133;
          v86 = v119 - 1;
        }
        v88 = v86 & (((unsigned int)v85 >> 9) ^ ((unsigned int)v85 >> 4));
        v89 = &v87[2 * v88];
        v90 = *v89;
        if ( v85 == *v89 )
          goto LABEL_114;
        v96 = 1;
        while ( v90 != -4096 )
        {
          v88 = v86 & (v96 + v88);
          v114 = v96 + 1;
          v89 = &v87[2 * v88];
          v90 = *v89;
          if ( v85 == *v89 )
            goto LABEL_114;
          v96 = v114;
        }
        if ( v84 )
        {
          v95 = 8;
          goto LABEL_134;
        }
        v94 = v119;
LABEL_133:
        v95 = 2 * v94;
LABEL_134:
        v89 = &v87[v95];
LABEL_114:
        v91 = 8;
        if ( !v84 )
          v91 = 2LL * v119;
        if ( v89 != &v87[v91]
          && *((_DWORD *)v89 + 3) > *((_DWORD *)v80 + 3)
          && *((_DWORD *)v89 + 2) < *((_DWORD *)v80 + 2) )
        {
          goto LABEL_60;
        }
        for ( v80 += 2; v81 != v80; v80 += 2 )
        {
          if ( *v80 != -8192 && *v80 != -4096 )
            break;
        }
        if ( v83 == v80 )
          goto LABEL_124;
      }
    }
    v29 = v113;
    do
    {
LABEL_55:
      v56 = *(_QWORD *)(v55 - 8);
      v55 -= 32;
      if ( v56 != 0 && v56 != -4096 && v56 != -8192 )
        sub_BD60C0((_QWORD *)(v55 + 8));
    }
    while ( v55 != v54 );
LABEL_59:
    *(_DWORD *)(a6 + 416) = 0;
    sub_2284740((__int64)&v120, (__int64)a3, v109);
    if ( v113 == v29 )
      goto LABEL_100;
LABEL_60:
    if ( *((_DWORD *)a2 + 2) <= i )
      break;
    if ( (v117 & 1) == 0 )
      sub_C7D6A0((__int64)v118, 16LL * v119, 8);
    v117 = 1;
    v57 = (__int64 *)&v118;
    do
    {
      *v57 = -4096;
      v57 += 2;
    }
    while ( v57 != (__int64 *)&v120 );
    v58 = v117 & 0xFFFFFFFE;
    LODWORD(v117) = v117 & 1 | v121 & 0xFFFFFFFE;
    v121 = v58 | v121 & 1;
    v59 = v122;
    v122 = HIDWORD(v117);
    HIDWORD(v117) = v59;
    if ( (v117 & 1) != 0 )
    {
      v61 = v116;
      v60 = (char *)&v120;
      if ( (v121 & 1) == 0 )
        goto LABEL_67;
      v67 = (__int64 *)&v118;
      v68 = (__int64 *)&v123;
      while ( 1 )
      {
        v69 = *v67;
        v70 = *v67 != -4096 && *v67 != -8192;
        v71 = *v68;
        if ( *v68 == -4096 )
        {
          *v67 = -4096;
          *v68 = v69;
          if ( v70 )
            goto LABEL_141;
        }
        else
        {
          if ( v71 != -8192 )
          {
            if ( v70 )
            {
              v75 = v67[1];
              *v67 = v71;
              v76 = v68[1];
              *v68 = v69;
              v68[1] = v75;
              v67[1] = v76;
            }
            else
            {
              *v68 = v69;
              v72 = v68[1];
              *v67 = v71;
              v67[1] = v72;
            }
            goto LABEL_86;
          }
          *v67 = -8192;
          *v68 = v69;
          if ( v70 )
LABEL_141:
            v68[1] = v67[1];
        }
LABEL_86:
        v67 += 2;
        v68 += 2;
        if ( v67 == (__int64 *)&v120 )
          goto LABEL_73;
      }
    }
    v60 = v116;
    v61 = (char *)&v120;
    if ( (v121 & 1) == 0 )
    {
      v74 = (__int64)v118;
      v73 = v119;
      v118 = v123;
      v123 = (__int64 *)v74;
      v119 = v124;
      v124 = v73;
      goto LABEL_89;
    }
LABEL_67:
    v60[8] |= 1u;
    v62 = *((_QWORD *)v60 + 2);
    v63 = (__int64 *)(v61 + 16);
    v64 = v60 + 24;
    v65 = *v64;
    do
    {
      v66 = *v63;
      *((_QWORD *)v64 - 1) = *v63;
      if ( v66 != -8192 && v66 != -4096 )
        *(_QWORD *)v64 = v63[1];
      v63 += 2;
      v64 += 4;
    }
    while ( v61 + 80 != (char *)v63 );
    v61[8] &= ~1u;
    *((_QWORD *)v61 + 2) = v62;
    *((_DWORD *)v61 + 6) = v65;
LABEL_73:
    if ( (v121 & 1) != 0 )
    {
      if ( !v129 )
        goto LABEL_90;
      goto LABEL_75;
    }
    v73 = v124;
    v74 = (__int64)v123;
LABEL_89:
    sub_C7D6A0(v74, 16LL * v73, 8);
    if ( !v129 )
    {
LABEL_90:
      _libc_free(v128);
      if ( v127 )
        continue;
      goto LABEL_91;
    }
LABEL_75:
    if ( v127 )
      continue;
LABEL_91:
    _libc_free(v126);
  }
  if ( byte_4FDAE68 )
    sub_C64ED0("Max devirtualization iterations reached", 1u);
  v92 = v121 & 1;
LABEL_125:
  if ( !v92 )
    sub_C7D6A0((__int64)v123, 16LL * v124, 8);
LABEL_127:
  sub_227AD40((__int64)v125);
  if ( (v117 & 1) == 0 )
    sub_C7D6A0((__int64)v118, 16LL * v119, 8);
  return a1;
}
