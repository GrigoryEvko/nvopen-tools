// Function: sub_EEFA10
// Address: 0xeefa10
//
__int64 __fastcall sub_EEFA10(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _BYTE *v6; // rax
  _BYTE *v7; // rdx
  __int64 v8; // r14
  _BYTE *v10; // rax
  char v12; // bl
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  _BYTE **v17; // rsi
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r9
  _QWORD *v21; // r8
  char v22; // al
  __int64 v23; // r14
  char *v24; // r15
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 v29; // r8
  __int64 v30; // r9
  __int64 v31; // rax
  __int64 v32; // rdx
  char *v33; // r14
  char *v34; // r12
  char *v35; // rbx
  unsigned __int64 v36; // r14
  unsigned __int64 v37; // r14
  __int64 v38; // rax
  _QWORD *v39; // rax
  __int64 v40; // r8
  __int64 *v41; // rax
  _QWORD *v42; // rax
  char *v43; // r12
  __int64 v44; // rdx
  __int64 v45; // rcx
  __int64 v46; // r8
  __int64 v47; // r9
  char v48; // bl
  __int64 v49; // rax
  __int64 v50; // r8
  _BYTE *v51; // rax
  __int64 v52; // r14
  unsigned __int64 *v53; // r12
  char v54; // al
  __int64 v55; // rdx
  __int64 v56; // rbx
  __int64 v57; // r8
  __int64 v58; // r9
  __int64 v59; // rcx
  __int64 v60; // r8
  __int64 v61; // r9
  __int64 v62; // rdx
  __int64 v63; // rcx
  __int64 v64; // r8
  __int64 v65; // r9
  _BYTE **v66; // rsi
  _QWORD *v67; // rax
  __int64 *v68; // rax
  __int64 v69; // rax
  _QWORD *v70; // rdi
  char *v71; // r15
  __int64 v72; // r14
  __int64 v73; // rax
  size_t v74; // rdx
  char *v75; // rcx
  char *v76; // rax
  char *v77; // rax
  __int64 v78; // rax
  __int64 v79; // rax
  __int64 v80; // rax
  size_t v81; // r8
  __int64 v82; // rcx
  char *v83; // rax
  char *v84; // rax
  __int64 v85; // rdi
  __int64 v86; // rax
  signed __int64 v87; // rdx
  signed __int64 v88; // rdx
  signed __int64 v89; // rax
  __int64 v90; // [rsp+8h] [rbp-108h]
  __int64 v91; // [rsp+10h] [rbp-100h]
  char v92; // [rsp+1Eh] [rbp-F2h]
  char v93; // [rsp+1Fh] [rbp-F1h]
  size_t n; // [rsp+20h] [rbp-F0h]
  size_t na; // [rsp+20h] [rbp-F0h]
  __int64 v96; // [rsp+28h] [rbp-E8h]
  char v97; // [rsp+28h] [rbp-E8h]
  __int64 v98; // [rsp+30h] [rbp-E0h]
  _QWORD *v99; // [rsp+38h] [rbp-D8h]
  _QWORD *v100; // [rsp+38h] [rbp-D8h]
  _QWORD *v101; // [rsp+38h] [rbp-D8h]
  _QWORD *v102; // [rsp+40h] [rbp-D0h] BYREF
  __int64 *v103; // [rsp+48h] [rbp-C8h] BYREF
  _BYTE *v104; // [rsp+50h] [rbp-C0h] BYREF
  __int64 v105; // [rsp+58h] [rbp-B8h]
  _BYTE v106[176]; // [rsp+60h] [rbp-B0h] BYREF

  v6 = *(_BYTE **)a1;
  v7 = *(_BYTE **)(a1 + 8);
  if ( v7 != *(_BYTE **)a1 && *v6 == 73 )
  {
    v10 = v6 + 1;
    v12 = a2;
    *(_QWORD *)a1 = v10;
    if ( !(_BYTE)a2 )
    {
LABEL_6:
      v98 = *(_QWORD *)(a1 + 24);
      v96 = *(_QWORD *)(a1 + 16);
      while ( 1 )
      {
        if ( v7 != v10 && *v10 == 69 )
        {
          v52 = 0;
          v50 = 0;
          *(_QWORD *)a1 = v10 + 1;
LABEL_40:
          n = v50;
          v53 = (unsigned __int64 *)sub_EE6060((_QWORD *)a1, (v98 - v96) >> 3);
          v54 = *(_BYTE *)(a1 + 937);
          v105 = 0x2000000000LL;
          v56 = v55;
          v103 = (__int64 *)&v104;
          v97 = v54;
          v104 = v106;
          sub_D953B0((__int64)&v104, 43, v55, 0x2000000000LL, v57, v58);
          sub_EE3CE0((__int64 *)&v103, v53, v56, v59, v60, v61);
          sub_D953B0((__int64)v103, v52, v62, v63, v64, v65);
          v66 = &v104;
          v67 = sub_C65B40(a1 + 904, (__int64)&v104, (__int64 *)&v103, (__int64)off_497B2F0);
          v8 = (__int64)v67;
          if ( v67 )
          {
            v8 = (__int64)(v67 + 1);
            if ( v104 != v106 )
              _libc_free(v104, &v104);
            v104 = (_BYTE *)v8;
            v68 = sub_EE6840(a1 + 944, (__int64 *)&v104);
            if ( v68 )
            {
              v69 = v68[1];
              if ( v69 )
                v8 = v69;
            }
            if ( *(_QWORD *)(a1 + 928) == v8 )
              *(_BYTE *)(a1 + 936) = 1;
          }
          else
          {
            if ( v97 )
            {
              v79 = sub_CD1D40((__int64 *)(a1 + 808), 48, 3);
              *(_QWORD *)v79 = 0;
              v66 = (_BYTE **)v79;
              v8 = v79 + 8;
              *(_WORD *)(v79 + 16) = 16427;
              LOBYTE(v79) = *(_BYTE *)(v79 + 18);
              v66[3] = v53;
              v66[4] = (_BYTE *)v56;
              v66[5] = (_BYTE *)n;
              *((_BYTE *)v66 + 18) = v79 & 0xF0 | 5;
              v66[1] = &unk_49DFDE8;
              sub_C657C0((__int64 *)(a1 + 904), (__int64 *)v66, v103, (__int64)off_497B2F0);
            }
            if ( v104 != v106 )
              _libc_free(v104, v66);
            *(_QWORD *)(a1 + 920) = v8;
          }
          return v8;
        }
        if ( !v12 )
        {
          v104 = sub_EEF530(a1);
          if ( !v104 )
            return 0;
          sub_E18380(a1 + 16, (__int64 *)&v104, v44, v45, v46, v47);
          goto LABEL_33;
        }
        v102 = sub_EEF530(a1);
        if ( !v102 )
          return 0;
        v17 = (_BYTE **)&v102;
        sub_E18380(a1 + 16, (__int64 *)&v102, v13, v14, v15, v16);
        v21 = v102;
        v22 = *((_BYTE *)v102 + 8);
        if ( v22 == 34 )
        {
          v21 = (_QWORD *)v102[3];
        }
        else if ( v22 == 41 )
        {
          v23 = v102[3];
          v24 = (char *)v102[2];
          v91 = v23;
          v92 = *(_BYTE *)(a1 + 937);
          v104 = v106;
          v105 = 0x2000000000LL;
          sub_D953B0((__int64)&v104, 40, v18, v19, (__int64)v102, v20);
          sub_D953B0((__int64)&v104, v23, v25, v26, v27, v28);
          v31 = 8 * v23;
          v32 = (unsigned int)v105;
          v33 = &v24[8 * v23];
          v90 = v31;
          if ( v24 != v33 )
          {
            v34 = v24;
            v93 = v12;
            v35 = v33;
            do
            {
              v36 = *(_QWORD *)v34;
              if ( v32 + 1 > (unsigned __int64)HIDWORD(v105) )
              {
                sub_C8D5F0((__int64)&v104, v106, v32 + 1, 4u, v29, v30);
                v32 = (unsigned int)v105;
              }
              *(_DWORD *)&v104[4 * v32] = v36;
              v37 = HIDWORD(v36);
              LODWORD(v105) = v105 + 1;
              v38 = (unsigned int)v105;
              if ( (unsigned __int64)(unsigned int)v105 + 1 > HIDWORD(v105) )
              {
                sub_C8D5F0((__int64)&v104, v106, (unsigned int)v105 + 1LL, 4u, v29, v30);
                v38 = (unsigned int)v105;
              }
              v34 += 8;
              *(_DWORD *)&v104[4 * v38] = v37;
              v32 = (unsigned int)(v105 + 1);
              LODWORD(v105) = v105 + 1;
            }
            while ( v35 != v34 );
            v33 = v35;
            v12 = v93;
          }
          v39 = sub_C65B40(a1 + 904, (__int64)&v104, (__int64 *)&v103, (__int64)off_497B2F0);
          if ( !v39 )
          {
            if ( !v92 )
            {
              if ( v104 != v106 )
                _libc_free(v104, &v104);
              *(_QWORD *)(a1 + 920) = 0;
              return 0;
            }
            v80 = sub_CD1D40((__int64 *)(a1 + 808), 40, 3);
            *(_QWORD *)v80 = 0;
            v17 = (_BYTE **)v80;
            v81 = v80 + 8;
            v82 = v90 >> 5;
            *(_QWORD *)(v80 + 24) = v24;
            *(_WORD *)(v80 + 16) = -32728;
            *(_QWORD *)(v80 + 8) = &unk_49DFCC8;
            *(_QWORD *)(v80 + 32) = v91;
            *(_BYTE *)(v80 + 18) = *(_BYTE *)(v80 + 18) & 0xF0 | 0xA;
            if ( v90 >> 5 > 0 )
            {
              v83 = v24;
              while ( (*(_BYTE *)(*(_QWORD *)v83 + 10LL) & 3) == 1 )
              {
                if ( (*(_BYTE *)(*((_QWORD *)v83 + 1) + 10LL) & 3) != 1 )
                {
                  v83 += 8;
                  break;
                }
                if ( (*(_BYTE *)(*((_QWORD *)v83 + 2) + 10LL) & 3) != 1 )
                {
                  v83 += 16;
                  break;
                }
                if ( (*(_BYTE *)(*((_QWORD *)v83 + 3) + 10LL) & 3) != 1 )
                {
                  v83 += 24;
                  break;
                }
                v83 += 32;
                if ( &v24[32 * v82] == v83 )
                  goto LABEL_100;
              }
              if ( v33 != v83 )
              {
LABEL_78:
                v84 = v24;
                v85 = v90 >> 5;
                while ( ((*(_BYTE *)(*(_QWORD *)v84 + 10LL) >> 2) & 3) == 1 )
                {
                  if ( ((*(_BYTE *)(*((_QWORD *)v84 + 1) + 10LL) >> 2) & 3) != 1 )
                  {
                    v84 += 8;
                    break;
                  }
                  if ( ((*(_BYTE *)(*((_QWORD *)v84 + 2) + 10LL) >> 2) & 3) != 1 )
                  {
                    v84 += 16;
                    break;
                  }
                  if ( ((*(_BYTE *)(*((_QWORD *)v84 + 3) + 10LL) >> 2) & 3) != 1 )
                  {
                    v84 += 24;
                    break;
                  }
                  v84 += 32;
                  if ( !--v85 )
                    goto LABEL_108;
                }
                if ( v84 != v33 )
                  goto LABEL_90;
LABEL_113:
                *((_BYTE *)v17 + 18) = *((_BYTE *)v17 + 18) & 0xF3 | 4;
LABEL_114:
                if ( v82 <= 0 )
                {
LABEL_115:
                  v89 = v33 - v24;
                  if ( v33 - v24 == 16 )
                  {
LABEL_139:
                    if ( *(_BYTE *)(*(_QWORD *)v24 + 9LL) >> 6 != 1 )
                      goto LABEL_91;
                    v24 += 8;
LABEL_118:
                    if ( *(_BYTE *)(*(_QWORD *)v24 + 9LL) >> 6 != 1 )
                      goto LABEL_91;
LABEL_119:
                    *((_BYTE *)v17 + 17) = *((_BYTE *)v17 + 17) & 0x3F | 0x40;
                  }
                  else
                  {
                    if ( v89 != 24 )
                    {
                      if ( v89 == 8 )
                        goto LABEL_118;
                      goto LABEL_119;
                    }
                    if ( *(_BYTE *)(*(_QWORD *)v24 + 9LL) >> 6 == 1 )
                    {
                      v24 += 8;
                      goto LABEL_139;
                    }
LABEL_91:
                    if ( v33 == v24 )
                      goto LABEL_119;
                  }
                  na = v81;
                  sub_C657C0((__int64 *)(a1 + 904), (__int64 *)v17, v103, (__int64)off_497B2F0);
                  v21 = (_QWORD *)na;
                  if ( v104 != v106 )
                  {
                    _libc_free(v104, v17);
                    v21 = (_QWORD *)na;
                  }
                  *(_QWORD *)(a1 + 920) = v21;
                  goto LABEL_29;
                }
LABEL_90:
                while ( *(_BYTE *)(*(_QWORD *)v24 + 9LL) >> 6 == 1 )
                {
                  if ( *(_BYTE *)(*((_QWORD *)v24 + 1) + 9LL) >> 6 != 1 )
                  {
                    v24 += 8;
                    goto LABEL_91;
                  }
                  if ( *(_BYTE *)(*((_QWORD *)v24 + 2) + 9LL) >> 6 != 1 )
                  {
                    v24 += 16;
                    goto LABEL_91;
                  }
                  if ( *(_BYTE *)(*((_QWORD *)v24 + 3) + 9LL) >> 6 != 1 )
                  {
                    v24 += 24;
                    goto LABEL_91;
                  }
                  v24 += 32;
                  if ( !--v82 )
                    goto LABEL_115;
                }
                goto LABEL_91;
              }
LABEL_105:
              *((_BYTE *)v17 + 18) = *((_BYTE *)v17 + 18) & 0xFC | 1;
LABEL_106:
              if ( v82 > 0 )
                goto LABEL_78;
              v84 = v24;
LABEL_108:
              v88 = v33 - v84;
              if ( v33 - v84 == 16 )
                goto LABEL_131;
              if ( v88 != 24 )
              {
                if ( v88 != 8 )
                  goto LABEL_113;
                goto LABEL_111;
              }
              if ( ((*(_BYTE *)(*(_QWORD *)v84 + 10LL) >> 2) & 3) == 1 )
              {
                v84 += 8;
LABEL_131:
                if ( ((*(_BYTE *)(*(_QWORD *)v84 + 10LL) >> 2) & 3) == 1 )
                {
                  v84 += 8;
LABEL_111:
                  if ( ((*(_BYTE *)(*(_QWORD *)v84 + 10LL) >> 2) & 3) == 1 )
                    goto LABEL_113;
                }
              }
              if ( v33 == v84 )
                goto LABEL_113;
              goto LABEL_114;
            }
            v83 = v24;
LABEL_100:
            v87 = v33 - v83;
            if ( v33 - v83 == 16 )
              goto LABEL_135;
            if ( v87 != 24 )
            {
              if ( v87 != 8 )
                goto LABEL_105;
              goto LABEL_103;
            }
            if ( (*(_BYTE *)(*(_QWORD *)v83 + 10LL) & 3) == 1 )
            {
              v83 += 8;
LABEL_135:
              if ( (*(_BYTE *)(*(_QWORD *)v83 + 10LL) & 3) == 1 )
              {
                v83 += 8;
LABEL_103:
                if ( (*(_BYTE *)(*(_QWORD *)v83 + 10LL) & 3) == 1 )
                  goto LABEL_105;
              }
            }
            if ( v33 == v83 )
              goto LABEL_105;
            goto LABEL_106;
          }
          v40 = (__int64)(v39 + 1);
          if ( v104 != v106 )
          {
            v99 = v39 + 1;
            _libc_free(v104, &v104);
            v40 = (__int64)v99;
          }
          v17 = &v104;
          v104 = (_BYTE *)v40;
          v100 = (_QWORD *)v40;
          v41 = sub_EE6840(a1 + 944, (__int64 *)&v104);
          v21 = v100;
          if ( v41 )
          {
            v42 = (_QWORD *)v41[1];
            if ( v42 )
              v21 = v42;
          }
          if ( *(_QWORD **)(a1 + 928) == v21 )
            *(_BYTE *)(a1 + 936) = 1;
        }
LABEL_29:
        v43 = *(char **)(a1 + 584);
        if ( v43 == *(char **)(a1 + 592) )
        {
          v71 = *(char **)(a1 + 576);
          v101 = v21;
          v72 = 16 * ((v43 - v71) >> 3);
          if ( v71 == (char *)(a1 + 600) )
          {
            v76 = (char *)malloc(16 * ((v43 - v71) >> 3), v17, v43 - v71, 16 * ((v43 - v71) >> 3), v21, v20);
            v75 = v76;
            if ( !v76 )
              goto LABEL_141;
            v21 = v101;
            v74 = v43 - v71;
            if ( v43 != v71 )
            {
              v77 = (char *)memmove(v76, v71, v74);
              v21 = v101;
              v74 = v43 - v71;
              v75 = v77;
            }
            *(_QWORD *)(a1 + 576) = v75;
          }
          else
          {
            v73 = realloc(v71);
            v21 = v101;
            v74 = v43 - v71;
            *(_QWORD *)(a1 + 576) = v73;
            v75 = (char *)v73;
            if ( !v73 )
              goto LABEL_141;
          }
          v43 = &v75[v74];
          *(_QWORD *)(a1 + 592) = &v75[v72];
        }
        *(_QWORD *)(a1 + 584) = v43 + 8;
        *(_QWORD *)v43 = v21;
LABEL_33:
        v10 = *(_BYTE **)a1;
        v7 = *(_BYTE **)(a1 + 8);
        if ( *(_BYTE **)a1 != v7 && *v10 == 81 )
        {
          v48 = *(_BYTE *)(a1 + 778);
          *(_BYTE *)(a1 + 778) = 1;
          *(_QWORD *)a1 = v10 + 1;
          v49 = sub_EEA9F0(a1);
          *(_BYTE *)(a1 + 778) = v48;
          v50 = v49;
          if ( !v49 )
            return 0;
          v51 = *(_BYTE **)a1;
          if ( *(_QWORD *)a1 == *(_QWORD *)(a1 + 8) || *v51 != 69 )
            return 0;
          v52 = v50;
          *(_QWORD *)a1 = v51 + 1;
          goto LABEL_40;
        }
      }
    }
    v70 = *(_QWORD **)(a1 + 664);
    *(_QWORD *)(a1 + 672) = v70;
    if ( v70 != *(_QWORD **)(a1 + 680) )
    {
LABEL_50:
      *(_QWORD *)(a1 + 672) = v70 + 1;
      *v70 = a1 + 576;
      v7 = *(_BYTE **)(a1 + 8);
      *(_QWORD *)(a1 + 584) = *(_QWORD *)(a1 + 576);
      v10 = *(_BYTE **)a1;
      goto LABEL_6;
    }
    if ( v70 == (_QWORD *)(a1 + 688) )
    {
      v86 = malloc(0, a2, v7, a4, a5, a6);
      v70 = (_QWORD *)v86;
      if ( v86 )
      {
        *(_QWORD *)(a1 + 664) = v86;
        goto LABEL_69;
      }
    }
    else
    {
      v78 = realloc(v70);
      *(_QWORD *)(a1 + 664) = v78;
      v70 = (_QWORD *)v78;
      if ( v78 )
      {
LABEL_69:
        *(_QWORD *)(a1 + 680) = v70;
        goto LABEL_50;
      }
    }
LABEL_141:
    abort();
  }
  return 0;
}
