// Function: sub_2B7A630
// Address: 0x2b7a630
//
__int64 ***__fastcall sub_2B7A630(__int64 a1, __int64 a2, int *a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r15d
  int *v8; // rbx
  bool v9; // zf
  size_t v10; // r13
  __int64 v11; // r8
  __int64 v12; // r8
  __int64 v13; // rdx
  unsigned __int64 v14; // rcx
  signed int v15; // esi
  __int64 v16; // rax
  __int64 v17; // r9
  unsigned __int64 *v18; // r13
  char v19; // r14
  unsigned int v20; // esi
  _QWORD *v21; // rax
  unsigned __int64 *v22; // r13
  __int64 v23; // rax
  __int64 **v24; // rax
  __int64 v25; // rax
  int *v26; // r13
  __int64 ***v27; // r12
  __int64 v28; // rax
  int v29; // r12d
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // r9
  __int64 v33; // r8
  __int64 v34; // rcx
  unsigned __int64 v35; // rax
  int v36; // edx
  __int64 v37; // rcx
  __int64 v38; // r8
  __int64 v39; // r15
  __int64 v40; // r9
  int *v41; // rdi
  char v42; // al
  __int64 v44; // rax
  char *v45; // rsi
  __int64 v46; // rcx
  __int64 v47; // rdx
  __int64 v48; // rcx
  __int64 v49; // r8
  __int64 v50; // r9
  __int64 v51; // rax
  __int64 v52; // rsi
  __int64 v53; // rcx
  __int64 v54; // rdx
  __int64 v55; // rax
  unsigned __int64 v56; // r15
  unsigned __int64 v57; // r15
  __int64 v58; // r9
  __int64 v59; // r12
  __int64 v60; // rax
  unsigned int v61; // r13d
  __int64 v62; // rax
  int v63; // edx
  void *v64; // rcx
  __int64 v65; // r8
  __int64 v66; // rbx
  __int64 *v67; // r14
  _BYTE *v68; // rax
  __int64 ***v69; // rax
  __int64 v70; // rsi
  _BYTE *v71; // rsi
  __int64 v72; // rax
  unsigned int v73; // edi
  __int64 v74; // rax
  __int64 v75; // rdx
  __int64 v76; // rcx
  __int64 v77; // r8
  __int64 v78; // r9
  __int64 v79; // rax
  _BYTE *v80; // rsi
  __int64 v81; // r8
  unsigned int v82; // edi
  __int64 v83; // rax
  __int64 v84; // rdx
  __int64 v85; // rcx
  __int64 v86; // r8
  __int64 v87; // r9
  char v88; // al
  __int64 ***v89; // rbx
  char v90; // al
  __int64 v91; // rax
  __int64 v92; // rax
  char v93; // [rsp+10h] [rbp-290h]
  __int64 v94; // [rsp+20h] [rbp-280h]
  unsigned __int64 v96; // [rsp+68h] [rbp-238h]
  __int64 i; // [rsp+70h] [rbp-230h]
  __int64 v98; // [rsp+70h] [rbp-230h]
  __int64 v99; // [rsp+78h] [rbp-228h]
  __int64 v100; // [rsp+78h] [rbp-228h]
  unsigned __int64 v101; // [rsp+78h] [rbp-228h]
  void *v102; // [rsp+78h] [rbp-228h]
  __int64 v103; // [rsp+80h] [rbp-220h] BYREF
  __int64 v104; // [rsp+88h] [rbp-218h] BYREF
  __int64 v105; // [rsp+90h] [rbp-210h] BYREF
  __int64 v106; // [rsp+98h] [rbp-208h] BYREF
  unsigned __int64 *v107; // [rsp+A0h] [rbp-200h] BYREF
  unsigned __int64 *v108; // [rsp+A8h] [rbp-1F8h] BYREF
  int *v109; // [rsp+B0h] [rbp-1F0h] BYREF
  __int64 v110; // [rsp+B8h] [rbp-1E8h]
  _BYTE v111[48]; // [rsp+C0h] [rbp-1E0h] BYREF
  void *s2; // [rsp+F0h] [rbp-1B0h] BYREF
  __int64 v113; // [rsp+F8h] [rbp-1A8h]
  _BYTE v114[48]; // [rsp+100h] [rbp-1A0h] BYREF
  _BYTE *v115; // [rsp+130h] [rbp-170h] BYREF
  __int64 v116; // [rsp+138h] [rbp-168h]
  _BYTE v117[48]; // [rsp+140h] [rbp-160h] BYREF
  _BYTE *v118; // [rsp+170h] [rbp-130h] BYREF
  __int64 v119; // [rsp+178h] [rbp-128h]
  _BYTE v120[48]; // [rsp+180h] [rbp-120h] BYREF
  __int64 ***v121; // [rsp+1B0h] [rbp-F0h] BYREF
  __int64 v122; // [rsp+1B8h] [rbp-E8h]
  _BYTE v123[48]; // [rsp+1C0h] [rbp-E0h] BYREF
  unsigned __int64 *v124; // [rsp+1F0h] [rbp-B0h] BYREF
  __int64 v125; // [rsp+1F8h] [rbp-A8h]
  _BYTE v126[48]; // [rsp+200h] [rbp-A0h] BYREF
  unsigned __int64 v127; // [rsp+230h] [rbp-70h] BYREF
  __int64 v128; // [rsp+238h] [rbp-68h]
  _BYTE v129[16]; // [rsp+240h] [rbp-60h] BYREF
  __int16 v130; // [rsp+250h] [rbp-50h]

  v6 = 1;
  v8 = a3;
  v9 = *(_BYTE *)(a6 + 8) == 17;
  v104 = a1;
  v103 = a2;
  v96 = a4;
  if ( v9 )
    v6 = *(_DWORD *)(a6 + 32);
  v10 = 4 * a4;
  v109 = (int *)v111;
  v11 = (4 * a4) >> 2;
  v110 = 0xC00000000LL;
  if ( (unsigned __int64)(4 * a4) > 0x30 )
  {
    v101 = (4 * a4) >> 2;
    sub_C8D5F0((__int64)&v109, v111, v101, 4u, v11, a6);
    v11 = v101;
    v41 = &v109[(unsigned int)v110];
  }
  else
  {
    if ( !v10 )
      goto LABEL_5;
    v41 = (int *)v111;
  }
  v100 = v11;
  memcpy(v41, a3, v10);
  v10 = (unsigned int)v110;
  v11 = v100;
LABEL_5:
  v12 = v10 + v11;
  LODWORD(v110) = v12;
  if ( v6 != 1 )
  {
    sub_2B312D0(v6, (__int64)&v109, (__int64)a3, a4, v12, a6);
    v8 = v109;
    v96 = (unsigned int)v110;
  }
  if ( !v103 )
  {
    v13 = v104;
    v14 = *(_QWORD *)(v104 + 8);
    v23 = v104;
    if ( *(_BYTE *)(v14 + 8) != 17 )
      goto LABEL_46;
    goto LABEL_45;
  }
  sub_2B79FB0((__int64 **)a5, &v104, &v103, a4, v12, a6);
  v13 = v104;
  v14 = v103;
  v15 = v96;
  v16 = *(_QWORD *)(v104 + 8);
  if ( *(_BYTE *)(v16 + 8) == 17 )
    v15 = *(_DWORD *)(v16 + 32);
  if ( !v103 )
  {
LABEL_45:
    v23 = v13;
    goto LABEL_46;
  }
  sub_2B23C00((__int64 *)&v124, v15, (__int64)v8, v96, 1);
  sub_2B25530(&v127, v103, (unsigned __int64 *)&v124);
  v18 = (unsigned __int64 *)v127;
  v19 = v127 & 1;
  if ( (v127 & 1) != 0 )
  {
    v14 = v127 >> 58;
    if ( (~(-1LL << (v127 >> 58)) & (v127 >> 1)) != (1LL << (v127 >> 58)) - 1 )
    {
      v22 = v124;
      if ( ((unsigned __int8)v124 & 1) != 0 || !v124 )
      {
        v23 = v104;
LABEL_33:
        v105 = v23;
        v28 = *(_QWORD *)(v23 + 8);
        v113 = 0xC00000000LL;
        v106 = v103;
        v29 = *(_DWORD *)(v28 + 32);
        s2 = v114;
        sub_11B1960((__int64)&s2, v96, -1, v14, v12, v17);
        v116 = 0xC00000000LL;
        v115 = v117;
        sub_11B1960((__int64)&v115, v96, -1, v30, v31, v32);
        v34 = (unsigned int)v96;
        if ( (int)v96 > 0 )
        {
          v34 = 4LL * (unsigned int)(v96 - 1) + 4;
          v35 = 0;
          do
          {
            while ( 1 )
            {
              v36 = v8[v35 / 4];
              if ( v36 >= v29 )
                break;
              *(_DWORD *)((char *)s2 + v35) = v36;
              v35 += 4LL;
              if ( v34 == v35 )
                goto LABEL_38;
            }
            *(_DWORD *)&v115[v35] = v36 - v29;
            v35 += 4LL;
          }
          while ( v34 != v35 );
        }
LABEL_38:
        v99 = v105;
        for ( i = v106; ; i = v40 )
        {
          while ( 1 )
          {
            sub_2B353C0(&v105, (__int64)&s2, 0, v34, v33);
            sub_2B353C0(&v106, (__int64)&v115, 0, v37, v38);
            v39 = v105;
            v40 = v106;
            if ( *(_BYTE *)v105 == 92 && *(_BYTE *)v106 == 92 )
            {
              v118 = v120;
              v94 = v106;
              v119 = 0xC00000000LL;
              sub_11B1960((__int64)&v118, v96, -1, v34, v33, v106);
              v44 = 0;
              v45 = (char *)s2;
              v46 = 4LL * (unsigned int)v113;
              if ( v46 )
              {
                do
                {
                  v47 = *(unsigned int *)&v45[v44];
                  if ( (_DWORD)v47 != -1 )
                    *(_DWORD *)&v118[v44] = *(_DWORD *)(*(_QWORD *)(v39 + 72) + 4 * v47);
                  v44 += 4;
                }
                while ( v46 != v44 );
              }
              sub_2B23C00(
                (__int64 *)&v107,
                *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v39 - 32) + 8LL) + 32LL),
                (__int64)v118,
                (unsigned int)v119,
                1);
              v121 = (__int64 ***)v123;
              v122 = 0xC00000000LL;
              sub_11B1960((__int64)&v121, (unsigned int)v116, -1, v48, v49, v50);
              v51 = 0;
              v52 = (__int64)v115;
              v53 = 4LL * (unsigned int)v116;
              if ( v53 )
              {
                do
                {
                  v54 = *(unsigned int *)(v52 + v51);
                  if ( (_DWORD)v54 != -1 )
                    *(_DWORD *)((char *)v121 + v51) = *(_DWORD *)(*(_QWORD *)(v94 + 72) + 4 * v54);
                  v51 += 4;
                }
                while ( v53 != v51 );
              }
              sub_2B23C00(
                (__int64 *)&v108,
                *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v94 - 32) + 8LL) + 32LL),
                (__int64)v121,
                (unsigned int)v122,
                1);
              v55 = *(_QWORD *)(*(_QWORD *)(v39 - 64) + 8LL);
              if ( *(_QWORD *)(*(_QWORD *)(v94 - 64) + 8LL) == v55 && v55 != *(_QWORD *)(v39 + 8) )
              {
                sub_2B25A00(&v124, *(char **)(v39 - 32), (unsigned __int64 *)&v107);
                if ( (unsigned __int8)sub_2B0D9E0((unsigned __int64)v124) )
                {
                  sub_2B25A00(&v127, *(char **)(v94 - 32), (unsigned __int64 *)&v108);
                  v93 = sub_2B0D9E0(v127);
                  sub_228BF40((unsigned __int64 **)&v127);
                  sub_228BF40(&v124);
                  if ( v93 )
                  {
                    v71 = *(_BYTE **)(v39 + 72);
                    v105 = *(_QWORD *)(v39 - 64);
                    v106 = *(_QWORD *)(v94 - 64);
                    v72 = *(unsigned int *)(v39 + 80);
                    v124 = (unsigned __int64 *)v126;
                    v125 = 0xC00000000LL;
                    sub_2B35330((__int64)&v124, v71, &v71[4 * v72], 0xC00000000LL, v33, v94);
                    v73 = v125;
                    v74 = *(_QWORD *)(v105 + 8);
                    if ( *(_BYTE *)(v74 + 8) == 17 )
                      v73 = *(_DWORD *)(v74 + 32);
                    sub_2B31420(v73, (__int64)&v124, (__int64)s2, (unsigned int)v113);
                    sub_2B310D0((__int64)&s2, (__int64)&v124, v75, v76, v77, v78);
                    v79 = *(unsigned int *)(v94 + 80);
                    v80 = *(_BYTE **)(v94 + 72);
                    v127 = (unsigned __int64)v129;
                    v128 = 0xC00000000LL;
                    sub_2B35330((__int64)&v127, v80, &v80[4 * v79], 0xC00000000LL, v81, v94);
                    v82 = v128;
                    v83 = *(_QWORD *)(v106 + 8);
                    if ( *(_BYTE *)(v83 + 8) == 17 )
                      v82 = *(_DWORD *)(v83 + 32);
                    sub_2B31420(v82, (__int64)&v127, (__int64)v115, (unsigned int)v116);
                    sub_2B310D0((__int64)&v115, (__int64)&v127, v84, v85, v86, v87);
                    if ( (_BYTE *)v127 != v129 )
                      _libc_free(v127);
                    if ( v124 != (unsigned __int64 *)v126 )
                      _libc_free((unsigned __int64)v124);
                  }
                }
                else
                {
                  sub_228BF40(&v124);
                }
              }
              v56 = (unsigned __int64)v108;
              if ( ((unsigned __int8)v108 & 1) == 0 && v108 )
              {
                if ( (unsigned __int64 *)*v108 != v108 + 2 )
                  _libc_free(*v108);
                j_j___libc_free_0(v56);
              }
              if ( v121 != (__int64 ***)v123 )
                _libc_free((unsigned __int64)v121);
              v57 = (unsigned __int64)v107;
              if ( ((unsigned __int8)v107 & 1) == 0 && v107 )
              {
                if ( (unsigned __int64 *)*v107 != v107 + 2 )
                  _libc_free(*v107);
                j_j___libc_free_0(v57);
              }
              if ( v118 != v120 )
                _libc_free((unsigned __int64)v118);
              v39 = v105;
              v40 = v106;
            }
            if ( v39 == v99 )
              break;
            i = v40;
            v99 = v39;
          }
          if ( v40 == i )
            break;
        }
        sub_2B79FB0((__int64 **)a5, &v105, &v106, v34, v33, v40);
        v58 = v106;
        v59 = v105;
        v60 = *(_QWORD *)(v105 + 8);
        v61 = *(_DWORD *)(*(_QWORD *)(v106 + 8) + 32LL);
        if ( *(_DWORD *)(v60 + 32) >= v61 )
          v61 = *(_DWORD *)(v60 + 32);
        if ( (int)v96 > 0 )
        {
          v62 = 0;
          do
          {
            v63 = *(_DWORD *)&v115[v62];
            if ( v63 != -1 )
            {
              if ( v59 != v58 )
                v63 += v61;
              *(_DWORD *)((char *)s2 + v62) = v63;
              v59 = v105;
              v58 = v106;
            }
            v62 += 4;
          }
          while ( 4LL * (unsigned int)(v96 - 1) + 4 != v62 );
        }
        if ( v58 == v59 )
        {
          v88 = sub_B4ED80((int *)s2, (unsigned int)v113, v61);
          v89 = (__int64 ***)v105;
          if ( v88
            || (v90 = sub_B4EE20((int *)s2, (unsigned int)v113, v61), v59 = v105, v90)
            && (v89 = (__int64 ***)v105, *(_BYTE *)v105 == 92)
            && (v92 = *(unsigned int *)(v105 + 80), v92 == (unsigned int)v113)
            && (!(4 * v92) || !memcmp(*(const void **)(v105 + 72), s2, 4 * v92)) )
          {
            v27 = v89;
            goto LABEL_92;
          }
          v58 = v106;
          v64 = s2;
          v65 = (unsigned int)v113;
          if ( v59 == v106 )
          {
            v98 = (unsigned int)v113;
            v102 = s2;
            v91 = sub_ACADE0(*(__int64 ***)(v59 + 8));
            v59 = v105;
            v65 = v98;
            v64 = v102;
            v58 = v91;
          }
        }
        else
        {
          v64 = s2;
          v65 = (unsigned int)v113;
        }
        v27 = (__int64 ***)sub_2B7A390(a5, v59, v58, v64, v65);
LABEL_92:
        if ( v115 != v117 )
          _libc_free((unsigned __int64)v115);
        if ( s2 != v114 )
          _libc_free((unsigned __int64)s2);
        goto LABEL_48;
      }
      goto LABEL_23;
    }
    v22 = v124;
    v19 = (unsigned __int8)v124 & 1;
    if ( ((unsigned __int8)v124 & 1) == 0 && v124 )
    {
LABEL_23:
      if ( (unsigned __int64 *)*v22 != v22 + 2 )
        _libc_free(*v22);
      j_j___libc_free_0((unsigned __int64)v22);
      goto LABEL_26;
    }
    v23 = v104;
LABEL_46:
    if ( *(_BYTE *)v23 == 13 )
      goto LABEL_28;
    goto LABEL_47;
  }
  v14 = *(unsigned int *)(v127 + 64);
  v20 = *(_DWORD *)(v127 + 64) >> 6;
  if ( v20 )
  {
    v21 = *(_QWORD **)v127;
    while ( *v21 == -1 )
    {
      if ( (_QWORD *)(*(_QWORD *)v127 + 8LL * (v20 - 1) + 8) == ++v21 )
        goto LABEL_100;
    }
    v19 = 1;
  }
  else
  {
LABEL_100:
    v14 &= 0x3Fu;
    if ( (_DWORD)v14 )
      v19 = *(_QWORD *)(*(_QWORD *)v127 + 8LL * v20) != (1LL << v14) - 1;
  }
  if ( v127 )
  {
    if ( *(_QWORD *)v127 != v127 + 16 )
      _libc_free(*(_QWORD *)v127);
    j_j___libc_free_0((unsigned __int64)v18);
  }
  v22 = v124;
  if ( ((unsigned __int8)v124 & 1) == 0 && v124 )
    goto LABEL_23;
LABEL_26:
  v23 = v104;
  if ( v19 )
    goto LABEL_33;
  if ( *(_BYTE *)v104 == 13 )
  {
LABEL_28:
    v24 = (__int64 **)sub_2B08680(*(_QWORD *)(*(_QWORD *)(v23 + 8) + 24LL), v96);
    v25 = sub_ACADE0(v24);
    v26 = v109;
    v27 = (__int64 ***)v25;
    goto LABEL_49;
  }
LABEL_47:
  v42 = sub_2B353C0(&v104, (__int64)&v109, 1, v14, v12);
  v27 = (__int64 ***)v104;
  if ( v42 )
  {
LABEL_48:
    v26 = v109;
    goto LABEL_49;
  }
  v66 = (unsigned int)v110;
  v26 = v109;
  v27 = (__int64 ***)v104;
  if ( (_DWORD)v110 )
  {
    if ( (_DWORD)v110 != *(_DWORD *)(*(_QWORD *)(v104 + 8) + 32LL)
      || !(unsigned __int8)sub_B4ED80(v109, (unsigned int)v110, v110) )
    {
      v67 = *(__int64 **)a5;
      v130 = 257;
      v68 = (_BYTE *)sub_ACADE0(v27[1]);
      v69 = (__int64 ***)sub_A83CB0((unsigned int **)v67, v27, v68, (__int64)v26, v66, (__int64)&v127);
      v27 = v69;
      if ( *(_BYTE *)v69 > 0x1Cu )
      {
        v121 = v69;
        sub_2B5C2D0(*(_QWORD *)(a5 + 8), (__int64 *)&v121);
        v70 = *(_QWORD *)(a5 + 16);
        v124 = (unsigned __int64 *)v121[5];
        sub_29B09C0((__int64)&v127, v70, (__int64 *)&v124);
      }
    }
    goto LABEL_48;
  }
LABEL_49:
  if ( v26 != (int *)v111 )
    _libc_free((unsigned __int64)v26);
  return v27;
}
