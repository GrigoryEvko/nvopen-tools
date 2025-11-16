// Function: sub_1028510
// Address: 0x1028510
//
__int64 __fastcall sub_1028510(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  __int64 v8; // rsi
  __int64 v9; // rax
  char v10; // dl
  __int64 v11; // r13
  __int64 v12; // rdx
  unsigned __int8 **v13; // r12
  __int64 v14; // r8
  unsigned __int64 v15; // rdx
  __int64 v16; // rcx
  char v17; // al
  _BYTE *v18; // r12
  unsigned __int64 v19; // r14
  __int64 v20; // rdx
  unsigned __int8 *v21; // rax
  unsigned __int8 *v22; // r12
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rax
  __int64 v29; // rdx
  unsigned int v30; // r15d
  unsigned int v31; // eax
  unsigned int v32; // r15d
  __int64 v33; // rdi
  __int64 (__fastcall *v34)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v35; // r9
  __int64 v36; // r15
  unsigned __int64 v37; // rax
  __int64 v38; // rsi
  unsigned __int8 *v39; // r15
  const char *v40; // rax
  __int64 v41; // rdi
  __int64 v42; // rdx
  __int64 (__fastcall *v43)(__int64, unsigned int, _BYTE *, unsigned __int8 *, unsigned __int8, char); // rax
  unsigned __int8 *v44; // r10
  const char *v45; // rax
  __int64 v46; // rdi
  __int64 v47; // r10
  __int64 v48; // rdx
  __int64 (__fastcall *v49)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *, unsigned __int8, char); // rax
  __int64 v50; // rax
  unsigned __int8 *v51; // r12
  unsigned int *v52; // r15
  __int64 v53; // r12
  __int64 v54; // rbx
  __int64 v55; // rdx
  bool v56; // zf
  __int64 v57; // rax
  unsigned int *v58; // rbx
  unsigned int *v59; // r15
  __int64 v60; // rdx
  __int64 v62; // r9
  __int64 v63; // r12
  int v64; // edx
  unsigned int v65; // ecx
  unsigned __int8 v66; // al
  __int64 v67; // rdx
  int v68; // r12d
  __int64 v69; // r12
  unsigned int *v70; // rbx
  unsigned int *v71; // r15
  __int64 v72; // r12
  __int64 v73; // rdx
  unsigned int v74; // esi
  __int64 v75; // rax
  __int64 v76; // rax
  unsigned __int8 *v77; // r15
  const char *v78; // rax
  __int64 v79; // rdi
  __int64 v80; // rdx
  __int64 (__fastcall *v81)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *, unsigned __int8, char); // rax
  unsigned __int8 *v82; // rax
  bool v83; // al
  __int64 *v84; // rax
  __int64 v85; // r12
  unsigned int *v86; // rbx
  unsigned int *v87; // r15
  __int64 v88; // r12
  __int64 v89; // rdx
  char v90; // [rsp+4h] [rbp-10Ch]
  __int64 v91; // [rsp+8h] [rbp-108h]
  __int64 v92; // [rsp+8h] [rbp-108h]
  __int64 v93; // [rsp+8h] [rbp-108h]
  __int64 v94; // [rsp+8h] [rbp-108h]
  __int64 v95; // [rsp+8h] [rbp-108h]
  __int64 v96; // [rsp+8h] [rbp-108h]
  unsigned __int8 v97; // [rsp+13h] [rbp-FDh]
  char v98; // [rsp+14h] [rbp-FCh]
  _BYTE **v99; // [rsp+18h] [rbp-F8h]
  __int64 v102; // [rsp+30h] [rbp-E0h]
  char v103; // [rsp+30h] [rbp-E0h]
  unsigned __int8 *v104; // [rsp+30h] [rbp-E0h]
  __int64 v105; // [rsp+30h] [rbp-E0h]
  unsigned __int8 *v106; // [rsp+30h] [rbp-E0h]
  unsigned __int8 *v107; // [rsp+30h] [rbp-E0h]
  unsigned __int8 **v108; // [rsp+38h] [rbp-D8h]
  unsigned __int8 *v109; // [rsp+40h] [rbp-D0h]
  __int64 v110; // [rsp+40h] [rbp-D0h]
  __int64 v111; // [rsp+40h] [rbp-D0h]
  unsigned __int8 *v112; // [rsp+40h] [rbp-D0h]
  unsigned __int8 *v113; // [rsp+40h] [rbp-D0h]
  _BYTE **v114; // [rsp+48h] [rbp-C8h]
  unsigned __int64 v115; // [rsp+50h] [rbp-C0h]
  signed __int64 v116; // [rsp+58h] [rbp-B8h]
  __int64 v117; // [rsp+60h] [rbp-B0h]
  __int64 v118; // [rsp+68h] [rbp-A8h]
  const char *v119; // [rsp+80h] [rbp-90h] BYREF
  __int64 v120; // [rsp+88h] [rbp-88h]
  char *v121; // [rsp+90h] [rbp-80h]
  __int16 v122; // [rsp+A0h] [rbp-70h]
  __int64 v123; // [rsp+B0h] [rbp-60h] BYREF
  __int64 v124; // [rsp+B8h] [rbp-58h]
  __int16 v125; // [rsp+D0h] [rbp-40h]

  v8 = *(_QWORD *)(a3 + 8);
  v9 = sub_AE4570(a2, v8);
  v10 = *(_BYTE *)(a3 + 1);
  v11 = v9;
  if ( (v10 & 4) != 0 && !a4 )
  {
    v98 = 1;
    v97 = (v10 & 8) != 0;
  }
  else
  {
    v98 = 0;
    v97 = (a4 | ((*(_BYTE *)(a3 + 1) >> 3) ^ 1) & 1) ^ 1;
  }
  if ( (*(_BYTE *)(a3 + 7) & 0x40) != 0 )
  {
    v12 = *(_QWORD *)(a3 - 8);
  }
  else
  {
    v8 = a3 - 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF);
    v12 = v8;
  }
  v13 = (unsigned __int8 **)(v12 + 32);
  v116 = sub_BB5290(a3) & 0xFFFFFFFFFFFFFFF9LL | 4;
  if ( (*(_BYTE *)(a3 + 7) & 0x40) != 0 )
  {
    v15 = *(_QWORD *)(a3 - 8);
    v99 = (_BYTE **)(v15 + 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF));
  }
  else
  {
    v99 = (_BYTE **)a3;
    v15 = a3 - 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF);
  }
  v114 = (_BYTE **)(v15 + 32);
  if ( (_BYTE **)(v15 + 32) == v99 )
    return sub_AD6530(v11, v8);
  v16 = v97;
  v108 = v13;
  v109 = 0;
  v17 = v97 | 2;
  if ( !v98 )
    v17 = v97;
  v90 = v17;
  do
  {
    v18 = *v114;
    v19 = v116 & 0xFFFFFFFFFFFFFFF8LL;
    v115 = v116 & 0xFFFFFFFFFFFFFFF8LL;
    if ( **v114 > 0x15u )
    {
      v102 = (v116 >> 1) & 3;
LABEL_29:
      v27 = *((_QWORD *)v18 + 1);
      if ( (unsigned int)*(unsigned __int8 *)(v11 + 8) - 17 <= 1 && (unsigned int)*(unsigned __int8 *)(v27 + 8) - 17 > 1 )
      {
        v125 = 257;
        LODWORD(v117) = *(_DWORD *)(v11 + 32);
        BYTE4(v117) = *(_BYTE *)(v11 + 8) == 18;
        v28 = sub_B37620((unsigned int **)a1, v117, (__int64)v18, &v123);
        v27 = *(_QWORD *)(v28 + 8);
        v18 = (_BYTE *)v28;
      }
      if ( v11 == v27 )
      {
LABEL_40:
        if ( v116 )
        {
          if ( v102 == 2 )
          {
            v36 = v116 & 0xFFFFFFFFFFFFFFF8LL;
            if ( v19 )
            {
LABEL_43:
              v103 = sub_AE5020(a2, v36);
              v123 = sub_9208B0(a2, v36);
              v8 = 1LL << v103;
              v124 = v20;
              v20 = (unsigned __int8)v20;
              v37 = (((unsigned __int64)(v123 + 7) >> 3) + (1LL << v103) - 1) >> v103 << v103;
LABEL_44:
              if ( v37 == 1 )
              {
                v44 = v18;
                if ( !(_BYTE)v20 )
                  goto LABEL_55;
              }
              v38 = v11;
              if ( (unsigned int)*(unsigned __int8 *)(v11 + 8) - 17 <= 1 )
                v38 = **(_QWORD **)(v11 + 16);
              v39 = (unsigned __int8 *)sub_B33F60(a1, v38, v37, v20);
              if ( (unsigned int)*(unsigned __int8 *)(v11 + 8) - 17 <= 1 )
              {
                v125 = 257;
                LODWORD(v118) = *(_DWORD *)(v11 + 32);
                BYTE4(v118) = *(_BYTE *)(v11 + 8) == 18;
                v39 = (unsigned __int8 *)sub_B37620((unsigned int **)a1, v118, (__int64)v39, &v123);
              }
              v40 = sub_BD5D20(a3);
              v122 = 773;
              v41 = *(_QWORD *)(a1 + 80);
              v119 = v40;
              v120 = v42;
              v121 = ".idx";
              v43 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, unsigned __int8 *, unsigned __int8, char))(*(_QWORD *)v41 + 32LL);
              if ( v43 == sub_9201A0 )
              {
                if ( *v18 > 0x15u || *v39 > 0x15u )
                {
LABEL_73:
                  v125 = 257;
                  v8 = sub_B504D0(17, (__int64)v18, (__int64)v39, (__int64)&v123, 0, 0);
                  (*(void (__fastcall **)(_QWORD, __int64, const char **, _QWORD, _QWORD))(**(_QWORD **)(a1 + 88) + 16LL))(
                    *(_QWORD *)(a1 + 88),
                    v8,
                    &v119,
                    *(_QWORD *)(a1 + 56),
                    *(_QWORD *)(a1 + 64));
                  v52 = *(unsigned int **)a1;
                  v44 = (unsigned __int8 *)v8;
                  v53 = *(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8);
                  if ( *(_QWORD *)a1 != v53 )
                  {
                    v105 = a1;
                    v54 = v8;
                    do
                    {
                      v55 = *((_QWORD *)v52 + 1);
                      v8 = *v52;
                      v52 += 4;
                      sub_B99FD0(v54, v8, v55);
                    }
                    while ( (unsigned int *)v53 != v52 );
                    v44 = (unsigned __int8 *)v54;
                    a1 = v105;
                  }
                  if ( v97 )
                  {
                    v8 = 1;
                    v107 = v44;
                    sub_B447F0(v44, 1);
                    v44 = v107;
                  }
                  if ( v98 )
                  {
                    v8 = 1;
                    v106 = v44;
                    sub_B44850(v44, 1);
                    v44 = v106;
                  }
LABEL_55:
                  if ( !v109 )
                  {
                    v109 = v44;
                    goto LABEL_22;
                  }
                  v104 = v44;
                  v45 = sub_BD5D20(a3);
                  v46 = *(_QWORD *)(a1 + 80);
                  v47 = (__int64)v104;
                  v119 = v45;
                  v122 = 773;
                  v120 = v48;
                  v121 = ".offs";
                  v49 = *(__int64 (__fastcall **)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *, unsigned __int8, char))(*(_QWORD *)v46 + 32LL);
                  if ( v49 == sub_9201A0 )
                  {
                    if ( *v109 > 0x15u || *v104 > 0x15u )
                      goto LABEL_89;
                    v8 = (__int64)v109;
                    if ( (unsigned __int8)sub_AC47B0(13) )
                      v50 = sub_AD5570(13, (__int64)v109, v104, v90, 0);
                    else
                      v50 = sub_AABE40(0xDu, v109, v104);
                    v47 = (__int64)v104;
                    v51 = (unsigned __int8 *)v50;
                  }
                  else
                  {
                    v8 = 13;
                    v75 = v49(v46, 13u, v109, v104, v97, v98 & 1);
                    v47 = (__int64)v104;
                    v51 = (unsigned __int8 *)v75;
                  }
                  if ( v51 )
                  {
LABEL_63:
                    v109 = v51;
                    goto LABEL_22;
                  }
LABEL_89:
                  v125 = 257;
                  v51 = (unsigned __int8 *)sub_B504D0(13, (__int64)v109, v47, (__int64)&v123, 0, 0);
                  v8 = (__int64)v51;
                  (*(void (__fastcall **)(_QWORD, unsigned __int8 *, const char **, _QWORD, _QWORD))(**(_QWORD **)(a1 + 88) + 16LL))(
                    *(_QWORD *)(a1 + 88),
                    v51,
                    &v119,
                    *(_QWORD *)(a1 + 56),
                    *(_QWORD *)(a1 + 64));
                  v57 = *(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8);
                  if ( *(_QWORD *)a1 != v57 )
                  {
                    v110 = a1;
                    v58 = *(unsigned int **)a1;
                    v59 = (unsigned int *)v57;
                    do
                    {
                      v60 = *((_QWORD *)v58 + 1);
                      v8 = *v58;
                      v58 += 4;
                      sub_B99FD0((__int64)v51, v8, v60);
                    }
                    while ( v59 != v58 );
                    a1 = v110;
                  }
                  if ( v97 )
                  {
                    v8 = 1;
                    sub_B447F0(v51, 1);
                  }
                  if ( v98 )
                  {
                    v8 = 1;
                    sub_B44850(v51, 1);
                  }
                  goto LABEL_63;
                }
                v8 = (__int64)v18;
                if ( (unsigned __int8)sub_AC47B0(17) )
                  v44 = (unsigned __int8 *)sub_AD5570(17, (__int64)v18, v39, v90, 0);
                else
                  v44 = (unsigned __int8 *)sub_AABE40(0x11u, v18, v39);
              }
              else
              {
                v8 = 17;
                v44 = (unsigned __int8 *)v43(v41, 17u, v18, v39, v97, v98 & 1);
              }
              if ( v44 )
                goto LABEL_55;
              goto LABEL_73;
            }
LABEL_86:
            v36 = sub_BCBAE0(v19, *v108, v27);
            goto LABEL_43;
          }
          if ( v102 != 1 )
            goto LABEL_86;
          if ( v19 )
            v36 = *(_QWORD *)(v19 + 24);
          else
            v36 = sub_BCBAE0(0, *v108, v27);
        }
        else
        {
          v36 = sub_BCBAE0(v19, *v108, v27);
          if ( v102 != 1 )
            goto LABEL_43;
        }
        v8 = v36;
        v123 = sub_9208B0(a2, v36);
        v124 = v20;
        v37 = (unsigned __int64)(v123 + 7) >> 3;
        v20 = (unsigned __int8)v20;
        goto LABEL_44;
      }
      v119 = sub_BD5D20((__int64)v18);
      v122 = 773;
      v120 = v29;
      v121 = ".c";
      v91 = *((_QWORD *)v18 + 1);
      v30 = sub_BCB060(v91);
      v31 = sub_BCB060(v11);
      v27 = v91;
      v32 = v31 < v30 ? 38 : 40;
      if ( v11 == v91 )
      {
        v35 = (__int64)v18;
        goto LABEL_39;
      }
      v33 = *(_QWORD *)(a1 + 80);
      v34 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v33 + 120LL);
      if ( v34 == sub_920130 )
      {
        if ( *v18 > 0x15u )
          goto LABEL_100;
        if ( (unsigned __int8)sub_AC4810(v32) )
          v35 = sub_ADAB70(v32, (unsigned __int64)v18, (__int64 **)v11, 0);
        else
          v35 = sub_AA93C0(v32, (unsigned __int64)v18, v11);
      }
      else
      {
        v35 = v34(v33, v32, v18, v11);
      }
      if ( v35 )
      {
LABEL_39:
        v18 = (_BYTE *)v35;
        goto LABEL_40;
      }
LABEL_100:
      v125 = 257;
      v62 = sub_B51D30(v32, (__int64)v18, v11, (__int64)&v123, 0, 0);
      if ( *(_BYTE *)v62 > 0x1Cu )
      {
        switch ( *(_BYTE *)v62 )
        {
          case ')':
          case '+':
          case '-':
          case '/':
          case '2':
          case '5':
          case 'J':
          case 'K':
          case 'S':
            goto LABEL_107;
          case 'T':
          case 'U':
          case 'V':
            v63 = *(_QWORD *)(v62 + 8);
            v64 = *(unsigned __int8 *)(v63 + 8);
            v65 = v64 - 17;
            v66 = *(_BYTE *)(v63 + 8);
            if ( (unsigned int)(v64 - 17) <= 1 )
              v66 = *(_BYTE *)(**(_QWORD **)(v63 + 16) + 8LL);
            if ( v66 <= 3u || v66 == 5 || (v66 & 0xFD) == 4 )
              goto LABEL_107;
            if ( (_BYTE)v64 == 15 )
            {
              if ( (*(_BYTE *)(v63 + 9) & 4) == 0 )
                break;
              v96 = v62;
              v83 = sub_BCB420(*(_QWORD *)(v62 + 8));
              v62 = v96;
              if ( !v83 )
                break;
              v84 = *(__int64 **)(v63 + 16);
              v63 = *v84;
              v64 = *(unsigned __int8 *)(*v84 + 8);
              v65 = v64 - 17;
            }
            else if ( (_BYTE)v64 == 16 )
            {
              do
              {
                v63 = *(_QWORD *)(v63 + 24);
                LOBYTE(v64) = *(_BYTE *)(v63 + 8);
              }
              while ( (_BYTE)v64 == 16 );
              v65 = (unsigned __int8)v64 - 17;
            }
            if ( v65 <= 1 )
              LOBYTE(v64) = *(_BYTE *)(**(_QWORD **)(v63 + 16) + 8LL);
            if ( (unsigned __int8)v64 <= 3u || (_BYTE)v64 == 5 || (v64 & 0xFD) == 4 )
            {
LABEL_107:
              v67 = *(_QWORD *)(a1 + 96);
              v68 = *(_DWORD *)(a1 + 104);
              if ( v67 )
              {
                v92 = v62;
                sub_B99FD0(v62, 3u, v67);
                v62 = v92;
              }
              v93 = v62;
              sub_B45150(v62, v68);
              v62 = v93;
            }
            break;
          default:
            break;
        }
      }
      v94 = v62;
      (*(void (__fastcall **)(_QWORD, __int64, const char **, _QWORD, _QWORD))(**(_QWORD **)(a1 + 88) + 16LL))(
        *(_QWORD *)(a1 + 88),
        v62,
        &v119,
        *(_QWORD *)(a1 + 56),
        *(_QWORD *)(a1 + 64));
      v35 = v94;
      v69 = *(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8);
      if ( *(_QWORD *)a1 != v69 )
      {
        v95 = a1;
        v70 = *(unsigned int **)a1;
        v71 = (unsigned int *)v69;
        v72 = v35;
        do
        {
          v73 = *((_QWORD *)v70 + 1);
          v74 = *v70;
          v70 += 4;
          sub_B99FD0(v72, v74, v73);
        }
        while ( v71 != v70 );
        a1 = v95;
        v35 = v72;
      }
      goto LABEL_39;
    }
    if ( sub_AD7890((__int64)v18, v8, v15, v16, v14) )
      goto LABEL_22;
    v102 = (v116 >> 1) & 3;
    if ( !v116 || ((v116 >> 1) & 3) != 0 || !v19 )
      goto LABEL_29;
    v21 = sub_AD8340(v18, v8, v20);
    if ( *((_DWORD *)v21 + 2) > 0x40u )
      v21 = *(unsigned __int8 **)v21;
    v22 = *(unsigned __int8 **)v21;
    v8 = v116 & 0xFFFFFFFFFFFFFFF8LL;
    v23 = 16LL * (unsigned int)v22 + sub_AE4AC0(a2, v19) + 24;
    v24 = *(_QWORD *)v23;
    LOBYTE(v23) = *(_BYTE *)(v23 + 8);
    v123 = v24;
    LOBYTE(v124) = v23;
    v25 = sub_CA1930(&v123);
    if ( v25 )
    {
      v8 = v25;
      v76 = sub_AD64C0(v11, v25, 0);
      v77 = (unsigned __int8 *)v76;
      if ( !v109 )
      {
        v109 = (unsigned __int8 *)v76;
        goto LABEL_22;
      }
      v78 = sub_BD5D20(a3);
      v79 = *(_QWORD *)(a1 + 80);
      v122 = 773;
      v119 = v78;
      v120 = v80;
      v121 = ".offs";
      v81 = *(__int64 (__fastcall **)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *, unsigned __int8, char))(*(_QWORD *)v79 + 32LL);
      if ( v81 == sub_9201A0 )
      {
        if ( *v109 > 0x15u || *v77 > 0x15u )
          goto LABEL_144;
        v8 = (__int64)v109;
        v82 = (unsigned __int8 *)((unsigned __int8)sub_AC47B0(13)
                                ? sub_AD5570(13, (__int64)v109, v77, v90, 0)
                                : sub_AABE40(0xDu, v109, v77));
LABEL_130:
        if ( !v82 )
        {
LABEL_144:
          v125 = 257;
          v8 = sub_B504D0(13, (__int64)v109, (__int64)v77, (__int64)&v123, 0, 0);
          (*(void (__fastcall **)(_QWORD, __int64, const char **, _QWORD, _QWORD))(**(_QWORD **)(a1 + 88) + 16LL))(
            *(_QWORD *)(a1 + 88),
            v8,
            &v119,
            *(_QWORD *)(a1 + 56),
            *(_QWORD *)(a1 + 64));
          v82 = (unsigned __int8 *)v8;
          v85 = *(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8);
          if ( *(_QWORD *)a1 != v85 )
          {
            v111 = a1;
            v86 = *(unsigned int **)a1;
            v87 = (unsigned int *)v85;
            v88 = v8;
            do
            {
              v89 = *((_QWORD *)v86 + 1);
              v8 = *v86;
              v86 += 4;
              sub_B99FD0(v88, v8, v89);
            }
            while ( v87 != v86 );
            a1 = v111;
            v82 = (unsigned __int8 *)v88;
          }
          if ( v97 )
          {
            v8 = 1;
            v113 = v82;
            sub_B447F0(v82, 1);
            v82 = v113;
          }
          if ( v98 )
          {
            v8 = 1;
            v112 = v82;
            sub_B44850(v82, 1);
            v82 = v112;
          }
        }
        v109 = v82;
        goto LABEL_22;
      }
      v8 = 13;
      v82 = (unsigned __int8 *)v81(v79, 13u, v109, v77, v97, v98 & 1);
      goto LABEL_130;
    }
LABEL_22:
    v114 += 4;
    if ( !v116 )
      goto LABEL_64;
    v26 = (v116 >> 1) & 3;
    if ( v26 == 2 )
    {
      if ( v19 )
        goto LABEL_25;
LABEL_64:
      v8 = (__int64)*v108;
      v115 = sub_BCBAE0(v19, *v108, v20);
      goto LABEL_25;
    }
    if ( v26 != 1 || !v19 )
      goto LABEL_64;
    v115 = *(_QWORD *)(v19 + 24);
LABEL_25:
    v15 = *(unsigned __int8 *)(v115 + 8);
    if ( (_BYTE)v15 == 16 )
    {
      v116 = *(_QWORD *)(v115 + 24) & 0xFFFFFFFFFFFFFFF9LL | 4;
    }
    else if ( (unsigned int)(unsigned __int8)v15 - 17 > 1 )
    {
      v56 = (_BYTE)v15 == 15;
      v15 = 0;
      if ( v56 )
        v15 = v115 & 0xFFFFFFFFFFFFFFF9LL;
      v116 = v15;
    }
    else
    {
      v116 = v115 & 0xFFFFFFFFFFFFFFF9LL | 2;
    }
    v108 += 4;
    v16 = (__int64)v99;
  }
  while ( v114 != v99 );
  if ( v109 )
    return (__int64)v109;
  return sub_AD6530(v11, v8);
}
