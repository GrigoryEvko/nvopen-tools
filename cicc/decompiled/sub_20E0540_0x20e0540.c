// Function: sub_20E0540
// Address: 0x20e0540
//
__int64 __fastcall sub_20E0540(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // r12
  __int64 (*v4)(); // rdx
  __int64 v5; // rax
  __int64 (*v6)(); // rax
  __int64 v7; // rax
  __int64 v8; // rdi
  __int64 (*v9)(); // rdx
  _QWORD *v10; // rax
  int v11; // r8d
  int v12; // r9d
  _QWORD *v13; // rax
  unsigned int v14; // r12d
  __int64 v15; // r13
  __int64 v16; // rax
  __int64 v17; // rdx
  _DWORD *v18; // rax
  _DWORD *i; // rdx
  __int64 v20; // r13
  __int64 v21; // r12
  __int64 v22; // rax
  __int64 v23; // rsi
  __int64 v24; // rax
  __int64 v25; // rax
  unsigned __int64 v26; // r13
  __int16 v27; // ax
  __int16 v28; // dx
  char v29; // r14
  __int64 v30; // r12
  char v31; // r12
  __int64 v32; // rax
  __int64 v33; // rdx
  unsigned __int64 v34; // r15
  bool v35; // al
  __int16 v36; // dx
  char v37; // r14
  unsigned __int8 v38; // r12
  __int16 v39; // dx
  __int16 v40; // cx
  bool v41; // al
  char v42; // r14
  bool v43; // al
  char v44; // r14
  __int64 v45; // rax
  __int64 v46; // r14
  __int64 v47; // r15
  __int64 v48; // rcx
  __int64 v49; // rax
  __int64 v50; // rax
  __int64 v51; // r8
  unsigned __int64 v52; // rdx
  __int64 v53; // rcx
  __int64 v54; // rdi
  unsigned __int64 v55; // r11
  __int64 v56; // rdi
  __int64 v57; // rdx
  unsigned __int64 v58; // rsi
  __int64 v59; // rax
  __int64 v60; // r8
  _QWORD *v61; // r9
  char *v62; // r13
  char *v63; // rsi
  __int64 v64; // rax
  __int64 v65; // rdi
  __int64 (*v66)(); // rax
  __int64 v67; // r13
  __int16 v68; // cx
  char v69; // r14
  unsigned __int64 v70; // rax
  bool v71; // al
  __int64 *v73; // rdi
  __int64 v74; // r12
  int v75; // r14d
  __int64 v76; // rax
  __int64 (*v77)(); // rdx
  unsigned __int64 v78; // r14
  __int64 *v79; // rbx
  _QWORD *v80; // r12
  _QWORD *j; // r15
  char *v82; // rsi
  _QWORD *v83; // rdx
  __int64 v84; // rsi
  int v85; // eax
  _QWORD *v86; // rdx
  _QWORD *v87; // r12
  unsigned __int64 v88; // rdi
  __int64 v89; // [rsp+0h] [rbp-B0h]
  __int64 v90; // [rsp+8h] [rbp-A8h]
  __int64 v91; // [rsp+10h] [rbp-A0h]
  __int64 *v92; // [rsp+20h] [rbp-90h]
  __int64 v93; // [rsp+28h] [rbp-88h]
  unsigned __int8 v94; // [rsp+30h] [rbp-80h]
  unsigned __int8 v95; // [rsp+3Fh] [rbp-71h]
  __int64 v96; // [rsp+40h] [rbp-70h]
  unsigned __int8 v97; // [rsp+48h] [rbp-68h]
  __int64 *v98; // [rsp+48h] [rbp-68h]
  __int64 v99; // [rsp+48h] [rbp-68h]
  __int64 v100; // [rsp+48h] [rbp-68h]
  __int64 v101; // [rsp+58h] [rbp-58h]
  __int64 v102; // [rsp+58h] [rbp-58h]
  __int64 v103; // [rsp+60h] [rbp-50h]
  __int64 v104; // [rsp+68h] [rbp-48h]
  __int64 v105[7]; // [rsp+78h] [rbp-38h] BYREF

  v2 = (__int64)a1;
  a1[57] = a2;
  v3 = *(_QWORD *)(a2 + 16);
  v4 = *(__int64 (**)())(*(_QWORD *)v3 + 40LL);
  v5 = 0;
  if ( v4 != sub_1D00B00 )
    v5 = ((__int64 (__fastcall *)(_QWORD))v4)(*(_QWORD *)(a2 + 16));
  a1[59] = v5;
  v6 = *(__int64 (**)())(*(_QWORD *)v3 + 112LL);
  if ( v6 == sub_1D00B10 )
  {
    a1[58] = 0;
    BUG();
  }
  v7 = ((__int64 (__fastcall *)(__int64))v6)(v3);
  a1[58] = v7;
  v8 = v7;
  v9 = *(__int64 (**)())(*(_QWORD *)v7 + 328LL);
  v10 = *(_QWORD **)(v2 + 456);
  if ( v9 != sub_1F49C90 )
  {
    if ( ((unsigned __int8 (__fastcall *)(__int64, _QWORD))v9)(v8, *(_QWORD *)(v2 + 456)) )
    {
      v86 = (_QWORD *)sub_22077B0(200);
      if ( v86 )
      {
        memset(v86, 0, 0xC8u);
        v86[6] = v86 + 8;
        v86[7] = 0x200000000LL;
      }
      v87 = *(_QWORD **)(v2 + 376);
      *(_QWORD *)(v2 + 376) = v86;
      if ( v87 )
      {
        _libc_free(v87[22]);
        _libc_free(v87[19]);
        _libc_free(v87[16]);
        _libc_free(v87[13]);
        v88 = v87[6];
        if ( (_QWORD *)v88 != v87 + 8 )
          _libc_free(v88);
        j_j___libc_free_0(v87, 200);
      }
    }
    v10 = *(_QWORD **)(v2 + 456);
  }
  sub_1E0BDD0(v10, 0);
  *(_DWORD *)(v2 + 240) = 0;
  v96 = v2 + 232;
  v13 = *(_QWORD **)(v2 + 456);
  v14 = (__int64)(v13[13] - v13[12]) >> 3;
  v15 = (__int64)(v13[13] - v13[12]) >> 3;
  if ( v14 )
  {
    v16 = 0;
    if ( v14 > (unsigned __int64)*(unsigned int *)(v2 + 244) )
    {
      sub_16CD150(v96, (const void *)(v2 + 248), v14, 8, v11, v12);
      v16 = 8LL * *(unsigned int *)(v2 + 240);
    }
    v17 = *(_QWORD *)(v2 + 232);
    v18 = (_DWORD *)(v17 + v16);
    for ( i = (_DWORD *)(v17 + 8LL * v14); i != v18; v18 += 2 )
    {
      if ( v18 )
      {
        *v18 = 0;
        v18[1] = 0;
      }
    }
    *(_DWORD *)(v2 + 240) = v15;
    v13 = *(_QWORD **)(v2 + 456);
  }
  v20 = v13[41];
  v21 = (__int64)(v13 + 40);
  if ( (_QWORD *)v20 != v13 + 40 )
  {
    do
    {
      *(_DWORD *)(*(_QWORD *)(v2 + 232) + 8LL * *(int *)(v20 + 48) + 4) = sub_20DF750(v2, v20);
      v20 = *(_QWORD *)(v20 + 8);
    }
    while ( v21 != v20 );
    v21 = *(_QWORD *)(*(_QWORD *)(v2 + 456) + 328LL);
  }
  sub_20DF590(v2, v21);
  v22 = *(_QWORD *)(v2 + 456);
  v95 = 0;
  v23 = v22 + 320;
  v24 = *(_QWORD *)(v22 + 328);
  v91 = v2 + 384;
  if ( v24 != v23 )
  {
    while ( 1 )
    {
      v103 = v24;
      v97 = 0;
      do
      {
        v26 = sub_1DD6160(v103);
        v104 = v103 + 24;
        if ( v26 != v103 + 24 )
        {
          v27 = *(_WORD *)(v26 + 46);
          v28 = v27 & 4;
          if ( (v27 & 4) != 0 || (v27 & 8) == 0 )
          {
            v29 = (unsigned __int8)*(_QWORD *)(*(_QWORD *)(v26 + 16) + 8LL) >> 7;
          }
          else
          {
            v29 = sub_1E15D00(v26, 0x80u, 1);
            v27 = *(_WORD *)(v26 + 46);
            v28 = v27 & 4;
          }
          if ( v28 || (v27 & 8) == 0 )
          {
            v30 = (*(_QWORD *)(*(_QWORD *)(v26 + 16) + 8LL) >> 5) & 1LL;
          }
          else
          {
            LOBYTE(v30) = sub_1E15D00(v26, 0x20u, 1);
            v27 = *(_WORD *)(v26 + 46);
            v28 = v27 & 4;
          }
          v31 = v29 & v30;
          if ( v28 || (v27 & 8) == 0 )
            v32 = (*(_QWORD *)(*(_QWORD *)(v26 + 16) + 8LL) >> 8) & 1LL;
          else
            LOBYTE(v32) = sub_1E15D00(v26, 0x100u, 1);
          v94 = v31 & (v32 ^ 1);
          if ( v94 )
          {
            v33 = (*(__int64 (__fastcall **)(_QWORD, unsigned __int64))(**(_QWORD **)(v2 + 472) + 248LL))(
                    *(_QWORD *)(v2 + 472),
                    v26);
            if ( v33 )
            {
              if ( !(unsigned __int8)sub_20DF6F0(v2, v26, v33) )
              {
                v73 = *(__int64 **)(v2 + 472);
                v74 = *(_QWORD *)(v26 + 24);
                v75 = -1;
                v76 = *v73;
                v77 = *(__int64 (**)())(*v73 + 128);
                if ( v77 != sub_1F39410 )
                {
                  v85 = ((__int64 (__fastcall *)(__int64 *, unsigned __int64))v77)(v73, v26);
                  v73 = *(__int64 **)(v2 + 472);
                  v75 = v85;
                  v76 = *v73;
                }
                v102 = (*(__int64 (__fastcall **)(__int64 *, unsigned __int64))(v76 + 248))(v73, v26);
                v100 = *(unsigned int *)(*(_QWORD *)(v2 + 232) + 8LL * *(int *)(v102 + 48));
                v93 = (unsigned int)sub_20DF640(v2, v26);
                *(_DWORD *)(*(_QWORD *)(v2 + 232) + 8LL * *(int *)(v74 + 48) + 4) -= v75;
                v78 = v74;
                if ( v74 + 24 != (*(_QWORD *)(v74 + 24) & 0xFFFFFFFFFFFFFFF8LL) )
                {
                  v78 = sub_20DFA10(v2, v74);
                  v92 = *(__int64 **)(v74 + 96);
                  if ( v92 != *(__int64 **)(v74 + 88) )
                  {
                    v90 = v74;
                    v89 = v2;
                    v79 = *(__int64 **)(v74 + 88);
                    do
                    {
                      v80 = *(_QWORD **)(*v79 + 160);
                      for ( j = (_QWORD *)sub_1DD77D0(*v79); v80 != j; *(_QWORD *)(v78 + 160) = v82 + 8 )
                      {
                        while ( 1 )
                        {
                          v82 = *(char **)(v78 + 160);
                          if ( v82 != *(char **)(v78 + 168) )
                            break;
                          v83 = j++;
                          sub_1DD8B80((char **)(v78 + 152), v82, v83);
                          if ( v80 == j )
                            goto LABEL_112;
                        }
                        if ( v82 )
                        {
                          *(_QWORD *)v82 = *j;
                          v82 = *(char **)(v78 + 160);
                        }
                        ++j;
                      }
LABEL_112:
                      ++v79;
                    }
                    while ( v92 != v79 );
                    v74 = v90;
                    v2 = v89;
                  }
                  sub_1DD6740(v78);
                  sub_1DD8FE0(v78, v102, -1);
                  sub_1DD9570(v74, v102, v78);
                }
                v84 = *(_QWORD *)(v26 + 64);
                v105[0] = v84;
                if ( v84 )
                  sub_1623A60((__int64)v105, v84, 2);
                sub_1E16240(v26);
                *(_DWORD *)(*(_QWORD *)(v2 + 232) + 8LL * *(int *)(v78 + 48) + 4) += (*(__int64 (__fastcall **)(_QWORD, unsigned __int64, __int64, __int64 *, __int64, _QWORD))(**(_QWORD **)(v2 + 472) + 256LL))(
                                                                                       *(_QWORD *)(v2 + 472),
                                                                                       v78,
                                                                                       v102,
                                                                                       v105,
                                                                                       v100 - v93,
                                                                                       *(_QWORD *)(v2 + 376));
                sub_20DF590(v2, v74);
                if ( v105[0] )
                  sub_161E7C0((__int64)v105, v105[0]);
                v97 = v94;
              }
            }
          }
          v34 = sub_1DD5EE0(v103);
          while ( v104 != v34 )
          {
            if ( !v34 )
              BUG();
            v36 = *(_WORD *)(v34 + 46);
            if ( (*(_BYTE *)v34 & 4) != 0 )
            {
              v67 = *(_QWORD *)(v34 + 8);
              v68 = *(_WORD *)(v34 + 46) & 4;
              if ( (v36 & 4) == 0 )
              {
                v68 = *(_WORD *)(v34 + 46) & 8;
                if ( (v36 & 8) != 0 )
                  goto LABEL_82;
              }
            }
            else if ( (v36 & 8) != 0 )
            {
              v70 = v34;
              do
                v70 = *(_QWORD *)(v70 + 8);
              while ( (*(_BYTE *)(v70 + 46) & 8) != 0 );
              v67 = *(_QWORD *)(v70 + 8);
              v68 = *(_WORD *)(v34 + 46) & 4;
              if ( (v36 & 4) == 0 )
              {
LABEL_82:
                v71 = sub_1E15D00(v34, 0x80u, 1);
                v36 = *(_WORD *)(v34 + 46);
                v69 = v71;
                v68 = v36 & 4;
                goto LABEL_74;
              }
            }
            else
            {
              v67 = *(_QWORD *)(v34 + 8);
              v68 = *(_WORD *)(v34 + 46) & 4;
            }
            v69 = (unsigned __int8)*(_QWORD *)(*(_QWORD *)(v34 + 16) + 8LL) >> 7;
LABEL_74:
            if ( v68 || (v36 & 8) == 0 )
            {
              v37 = ((*(_QWORD *)(*(_QWORD *)(v34 + 16) + 8LL) & 0x20LL) == 0) & v69;
              if ( v68 )
                goto LABEL_76;
            }
            else
            {
              v35 = sub_1E15D00(v34, 0x20u, 1);
              v36 = *(_WORD *)(v34 + 46);
              v37 = !v35 & v69;
              if ( (v36 & 4) != 0 )
                goto LABEL_76;
            }
            if ( (v36 & 8) != 0 )
            {
              v38 = v37 & !sub_1E15D00(v34, 0x100u, 1);
              if ( !v38 )
                goto LABEL_77;
              goto LABEL_38;
            }
LABEL_76:
            v38 = v37 & ((*(_QWORD *)(*(_QWORD *)(v34 + 16) + 8LL) & 0x100LL) == 0);
            if ( !v38 )
            {
LABEL_77:
              v34 = v67;
              continue;
            }
LABEL_38:
            v101 = (*(__int64 (__fastcall **)(_QWORD, unsigned __int64))(**(_QWORD **)(v2 + 472) + 248LL))(
                     *(_QWORD *)(v2 + 472),
                     v34);
            if ( (unsigned __int8)sub_20DF6F0(v2, v34, v101) )
              goto LABEL_77;
            if ( v104 == v67 )
              goto LABEL_83;
            v39 = *(_WORD *)(v67 + 46);
            v40 = v39 & 4;
            if ( (v39 & 4) != 0 || (v39 & 8) == 0 )
            {
              v42 = (unsigned __int8)*(_QWORD *)(*(_QWORD *)(v67 + 16) + 8LL) >> 7;
              if ( (v39 & 4) != 0 )
                goto LABEL_85;
            }
            else
            {
              v41 = sub_1E15D00(v67, 0x80u, 1);
              v39 = *(_WORD *)(v67 + 46);
              v42 = v41;
              v40 = v39 & 4;
              if ( (v39 & 4) != 0 )
                goto LABEL_85;
            }
            if ( (v39 & 8) != 0 )
            {
              v43 = sub_1E15D00(v67, 0x20u, 1);
              v39 = *(_WORD *)(v67 + 46);
              v44 = !v43 & v42;
              if ( (v39 & 4) != 0 )
                goto LABEL_86;
              goto LABEL_45;
            }
LABEL_85:
            v44 = ((*(_QWORD *)(*(_QWORD *)(v67 + 16) + 8LL) & 0x20LL) == 0) & v42;
            if ( v40 )
            {
LABEL_86:
              v45 = (*(_QWORD *)(*(_QWORD *)(v67 + 16) + 8LL) >> 8) & 1LL;
              goto LABEL_47;
            }
LABEL_45:
            if ( (v39 & 8) == 0 )
              goto LABEL_86;
            LOBYTE(v45) = sub_1E15D00(v67, 0x100u, 1);
LABEL_47:
            if ( (_BYTE)v45 != 1 && v44 )
            {
              v46 = *(_QWORD *)(v67 + 24);
              v47 = (__int64)sub_1E0B6F0(*(_QWORD *)(v2 + 456), *(_QWORD *)(v46 + 40));
              v98 = *(__int64 **)(v46 + 8);
              sub_1DD8DC0(*(_QWORD *)(v2 + 456) + 320LL, v47);
              v48 = *v98;
              v49 = *(_QWORD *)v47 & 7LL;
              *(_QWORD *)(v47 + 8) = v98;
              v48 &= 0xFFFFFFFFFFFFFFF8LL;
              *(_QWORD *)v47 = v48 | v49;
              *(_QWORD *)(v48 + 8) = v47;
              *v98 = v47 | *v98 & 7;
              v50 = v46 + 24;
              if ( v46 + 24 != v67 )
              {
                v51 = v47 + 24;
                if ( v50 != v47 + 24 )
                {
                  if ( v47 + 16 != v46 + 16 )
                  {
                    sub_1DD5C00((__int64 *)(v47 + 16), v46 + 16, v67, v46 + 24);
                    v51 = v47 + 24;
                    v50 = v46 + 24;
                  }
                  v52 = *(_QWORD *)(v46 + 24) & 0xFFFFFFFFFFFFFFF8LL;
                  *(_QWORD *)((*(_QWORD *)v67 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v50;
                  *(_QWORD *)(v46 + 24) = *(_QWORD *)(v46 + 24) & 7LL | *(_QWORD *)v67 & 0xFFFFFFFFFFFFFFF8LL;
                  v53 = *(_QWORD *)(v47 + 24);
                  *(_QWORD *)(v52 + 8) = v51;
                  v53 &= 0xFFFFFFFFFFFFFFF8LL;
                  *(_QWORD *)v67 = v53 | *(_QWORD *)v67 & 7LL;
                  *(_QWORD *)(v53 + 8) = v67;
                  *(_QWORD *)(v47 + 24) = v52 | *(_QWORD *)(v47 + 24) & 7LL;
                }
              }
              v54 = *(_QWORD *)(v2 + 472);
              v105[0] = 0;
              (*(void (__fastcall **)(__int64, __int64, __int64, _QWORD, _QWORD, _QWORD, __int64 *, _QWORD))(*(_QWORD *)v54 + 288LL))(
                v54,
                v46,
                v47,
                0,
                0,
                0,
                v105,
                0);
              if ( v105[0] )
                sub_161E7C0((__int64)v105, v105[0]);
              v55 = *(unsigned int *)(v2 + 240);
              v105[0] = 0;
              v56 = *(_QWORD *)(v2 + 232);
              v57 = 8 * v55;
              v58 = *(unsigned int *)(v2 + 244);
              LODWORD(v59) = v55;
              v60 = 8LL * *(int *)(v47 + 48);
              v61 = (_QWORD *)(v56 + 8 * v55);
              v62 = (char *)(v56 + v60);
              if ( (_QWORD *)(v56 + v60) == v61 )
              {
                if ( (unsigned int)v58 <= (unsigned int)v55 )
                {
                  sub_16CD150(v96, (const void *)(v2 + 248), 0, 8, v60, (int)v61);
                  v61 = (_QWORD *)(*(_QWORD *)(v2 + 232) + 8LL * *(unsigned int *)(v2 + 240));
                }
                *v61 = 0;
                ++*(_DWORD *)(v2 + 240);
              }
              else
              {
                if ( v55 >= v58 )
                {
                  v99 = 8LL * *(int *)(v47 + 48);
                  sub_16CD150(v96, (const void *)(v2 + 248), 0, 8, v60, (int)v61);
                  v56 = *(_QWORD *)(v2 + 232);
                  v59 = *(unsigned int *)(v2 + 240);
                  v57 = 8 * v59;
                  v62 = (char *)(v56 + v99);
                  v61 = (_QWORD *)(v56 + 8 * v59);
                }
                v63 = (char *)(v56 + v57 - 8);
                if ( v61 )
                {
                  *v61 = *(_QWORD *)v63;
                  v56 = *(_QWORD *)(v2 + 232);
                  v59 = *(unsigned int *)(v2 + 240);
                  v57 = 8 * v59;
                  v63 = (char *)(v56 + 8 * v59 - 8);
                }
                if ( v62 != v63 )
                {
                  memmove((void *)(v57 - (v63 - v62) + v56), v62, v63 - v62);
                  LODWORD(v59) = *(_DWORD *)(v2 + 240);
                }
                v64 = (unsigned int)(v59 + 1);
                *(_DWORD *)(v2 + 240) = v64;
                if ( v62 > (char *)v105 || (unsigned __int64)v105 >= *(_QWORD *)(v2 + 232) + 8 * v64 )
                  *(_QWORD *)v62 = v105[0];
                else
                  *(_QWORD *)v62 = v105[1];
              }
              sub_1DD91F0(v47, (_QWORD *)v46);
              sub_1DD8FE0(v46, v47, -1);
              sub_1DD8FE0(v46, v101, -1);
              sub_1DD7120((__int64 *)v47);
              sub_1DD7120((__int64 *)v46);
              *(_DWORD *)(*(_QWORD *)(v2 + 232) + 8LL * *(int *)(v46 + 48) + 4) = sub_20DF750(v2, v46);
              *(_DWORD *)(*(_QWORD *)(v2 + 232) + 8LL * *(int *)(v47 + 48) + 4) = sub_20DF750(v2, v47);
              sub_20DF590(v2, v46);
              v65 = *(_QWORD *)(v2 + 464);
              v66 = *(__int64 (**)())(*(_QWORD *)v65 + 328LL);
              if ( v66 != sub_1F49C90
                && ((unsigned __int8 (__fastcall *)(__int64, _QWORD))v66)(v65, *(_QWORD *)(v2 + 456)) )
              {
                sub_1DC3250(v91, (_QWORD *)v47);
              }
              goto LABEL_67;
            }
LABEL_83:
            sub_20DFEC0((_QWORD *)v2, v34);
LABEL_67:
            v97 = v38;
            v34 = sub_1DD5EE0(v103);
          }
        }
        v103 = *(_QWORD *)(v103 + 8);
        v25 = *(_QWORD *)(v2 + 456);
      }
      while ( v103 != v25 + 320 );
      if ( v97 )
      {
        v95 = v97;
        v24 = *(_QWORD *)(v25 + 328);
        if ( v24 != v103 )
          continue;
      }
      break;
    }
  }
  *(_DWORD *)(v2 + 240) = 0;
  return v95;
}
