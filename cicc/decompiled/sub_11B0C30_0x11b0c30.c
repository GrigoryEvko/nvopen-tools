// Function: sub_11B0C30
// Address: 0x11b0c30
//
__int64 __fastcall sub_11B0C30(unsigned __int8 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _DWORD *v8; // r12
  __int64 v9; // r14
  int v10; // eax
  _BYTE *v11; // rdi
  __int64 v12; // rax
  __int64 result; // rax
  __int64 v14; // rax
  bool v15; // r14
  unsigned __int8 v16; // dl
  __int64 v17; // rax
  unsigned __int8 *v18; // r13
  char v19; // r15
  __int64 v20; // rbx
  __int64 v21; // r14
  unsigned __int8 *v22; // rdx
  __int64 v23; // r12
  __int64 v24; // rax
  unsigned __int64 v25; // rdx
  unsigned int v26; // r11d
  unsigned __int8 *v27; // rcx
  bool v28; // al
  char v29; // r14
  __int64 *v30; // r12
  __int64 v31; // r13
  __int64 v32; // rbx
  unsigned int v33; // r14d
  int v34; // edx
  __int64 v35; // r14
  __int64 v36; // r12
  unsigned __int8 *v37; // r10
  __int64 v38; // r13
  __int64 v39; // rax
  __int64 v40; // r14
  _QWORD *v41; // rax
  __int64 v42; // r9
  __int64 v43; // rbx
  __int64 v44; // rax
  __int64 v45; // rbx
  __int64 v46; // r12
  __int64 v47; // r12
  __int64 v48; // rdx
  unsigned int v49; // esi
  int v50; // eax
  unsigned int v51; // r14d
  __int64 v52; // rbx
  __int64 v53; // r12
  unsigned __int8 *v54; // r13
  unsigned __int64 v55; // rax
  __int64 v56; // rdx
  char v57; // al
  unsigned __int8 v58; // al
  __int64 v59; // rdi
  __int64 v60; // rax
  __int64 v61; // rax
  __int64 v62; // r13
  __int64 v63; // r12
  unsigned int v64; // r14d
  __int64 v65; // rdx
  int v66; // r12d
  unsigned int *v67; // rbx
  __int64 v68; // r12
  __int64 v69; // rdx
  __int64 v70; // rax
  __int64 v71; // r13
  __int64 *v72; // r12
  int v73; // ebx
  __int64 v74; // r14
  unsigned int v75; // ebx
  __int64 v76; // r10
  unsigned int v77; // ecx
  unsigned int *v78; // rbx
  __int64 v79; // r12
  __int64 v80; // rdx
  __int16 v81; // si
  __int64 v82; // rcx
  __int64 v83; // rdx
  __int64 v84; // r13
  __int64 v85; // rbx
  __int64 v86; // rdx
  __int16 v87; // r12
  _QWORD **v88; // rdx
  int v89; // ecx
  __int64 *v90; // rax
  __int64 v91; // rsi
  unsigned int *v92; // rbx
  __int64 v93; // r12
  __int64 v94; // rdx
  __int64 **v95; // rax
  __int64 **v96; // rax
  __int64 **v97; // rax
  __int64 v98; // rdx
  int v99; // r12d
  unsigned int *v100; // rbx
  __int64 v101; // r12
  __int64 v102; // rdx
  __int64 *v103; // rsi
  __int64 *v104; // rax
  __int64 v105; // rdx
  int v106; // edi
  char v107; // al
  __int64 v108; // rax
  char v109; // [rsp+8h] [rbp-148h]
  __int64 v110; // [rsp+10h] [rbp-140h]
  _QWORD *v111; // [rsp+10h] [rbp-140h]
  unsigned int v112; // [rsp+10h] [rbp-140h]
  __int64 v113; // [rsp+10h] [rbp-140h]
  _DWORD *v114; // [rsp+20h] [rbp-130h]
  unsigned int v115; // [rsp+20h] [rbp-130h]
  __int64 v117; // [rsp+28h] [rbp-128h]
  __int64 v118; // [rsp+30h] [rbp-120h]
  __int64 v119; // [rsp+38h] [rbp-118h]
  _QWORD v120[4]; // [rsp+40h] [rbp-110h] BYREF
  __int16 v121; // [rsp+60h] [rbp-F0h]
  _BYTE *v122; // [rsp+70h] [rbp-E0h] BYREF
  __int64 v123; // [rsp+78h] [rbp-D8h]
  _BYTE v124[16]; // [rsp+80h] [rbp-D0h] BYREF
  __int16 v125; // [rsp+90h] [rbp-C0h]
  __int64 *v126; // [rsp+C0h] [rbp-90h] BYREF
  __int64 v127; // [rsp+C8h] [rbp-88h]
  __int64 v128; // [rsp+D0h] [rbp-80h] BYREF
  int v129; // [rsp+D8h] [rbp-78h]
  char v130; // [rsp+DCh] [rbp-74h]
  __int16 v131; // [rsp+E0h] [rbp-70h] BYREF

  v8 = (_DWORD *)a2;
  while ( 2 )
  {
    v9 = *((_QWORD *)a1 + 1);
    if ( (unsigned int)*(unsigned __int8 *)(v9 + 8) - 17 <= 1 )
      v9 = **(_QWORD **)(v9 + 16);
    v10 = *a1;
    if ( (_BYTE)v10 == 13 )
    {
      v95 = (__int64 **)sub_BCDA70((__int64 *)v9, a3);
      return sub_ACADE0(v95);
    }
    if ( (unsigned __int8)(v10 - 12) <= 1u )
      goto LABEL_102;
    if ( (unsigned __int8)(v10 - 9) <= 2u )
    {
      a2 = (__int64)a1;
      v126 = 0;
      v122 = v124;
      v123 = 0x800000000LL;
      v127 = (__int64)&v131;
      v120[0] = &v126;
      v128 = 8;
      v129 = 0;
      v130 = 1;
      v120[1] = &v122;
      v109 = sub_AA8FD0(v120, (__int64)a1);
      if ( v109 )
      {
        while ( 1 )
        {
          v11 = v122;
          if ( !(_DWORD)v123 )
            break;
          a2 = *(_QWORD *)&v122[8 * (unsigned int)v123 - 8];
          LODWORD(v123) = v123 - 1;
          if ( !(unsigned __int8)sub_AA8FD0(v120, a2) )
            goto LABEL_57;
        }
      }
      else
      {
LABEL_57:
        v109 = 0;
        v11 = v122;
      }
      if ( v11 != v124 )
        _libc_free(v11, a2);
      if ( !v130 )
        _libc_free(v127, a2);
      if ( v109 )
      {
LABEL_102:
        v96 = (__int64 **)sub_BCDA70((__int64 *)v9, a3);
        return sub_ACA8A0(v96);
      }
      v10 = *a1;
    }
    if ( (_BYTE)v10 == 14 )
    {
      v97 = (__int64 **)sub_BCDA70((__int64 *)v9, a3);
      return sub_AC9350(v97);
    }
    if ( (unsigned __int8)v10 <= 0x15u )
    {
      v12 = sub_ACADE0(*((__int64 ***)a1 + 1));
      return sub_AD5CE0((__int64)a1, v12, v8, a3, 0);
    }
    switch ( v10 )
    {
      case '*':
      case '+':
      case ',':
      case '-':
      case '.':
      case '/':
      case '0':
      case '1':
      case '2':
      case '3':
      case '4':
      case '5':
      case '6':
      case '7':
      case '8':
      case '9':
      case ':':
      case ';':
      case '?':
      case 'C':
      case 'D':
      case 'E':
      case 'F':
      case 'G':
      case 'H':
      case 'I':
      case 'J':
      case 'K':
      case 'R':
      case 'S':
      case 'V':
        v126 = &v128;
        v127 = 0x800000000LL;
        v14 = *(unsigned int *)(*((_QWORD *)a1 + 1) + 32LL);
        v15 = v14 != a3;
        if ( (*((_DWORD *)a1 + 1) & 0x7FFFFFF) == 0 )
        {
          if ( v14 == a3 )
            return (__int64)a1;
          v30 = &v128;
          v31 = 0;
          v26 = 0;
          goto LABEL_63;
        }
        v16 = a1[7];
        v114 = v8;
        v17 = a3;
        v18 = a1;
        v110 = 32LL * (*((_DWORD *)a1 + 1) & 0x7FFFFFF);
        v19 = v15;
        v20 = 0;
        v21 = v17;
        do
        {
          if ( (v16 & 0x40) != 0 )
            v22 = (unsigned __int8 *)*((_QWORD *)v18 - 1);
          else
            v22 = &v18[-32 * (*((_DWORD *)v18 + 1) & 0x7FFFFFF)];
          v23 = *(_QWORD *)&v22[v20];
          if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v23 + 8) + 8LL) - 17 <= 1 )
          {
            a2 = (__int64)v114;
            v23 = sub_11B0C30(*(_QWORD *)&v22[v20], v114, v21, a4);
          }
          v24 = (unsigned int)v127;
          v25 = (unsigned int)v127 + 1LL;
          if ( v25 > HIDWORD(v127) )
          {
            a2 = (__int64)&v128;
            sub_C8D5F0((__int64)&v126, &v128, v25, 8u, a5, a6);
            v24 = (unsigned int)v127;
          }
          v126[v24] = v23;
          v26 = v127 + 1;
          LODWORD(v127) = v127 + 1;
          v16 = v18[7];
          if ( (v16 & 0x40) != 0 )
            v27 = (unsigned __int8 *)*((_QWORD *)v18 - 1);
          else
            v27 = &v18[-32 * (*((_DWORD *)v18 + 1) & 0x7FFFFFF)];
          v28 = *(_QWORD *)&v27[v20] != v23;
          v20 += 32;
          v19 |= v28;
        }
        while ( v110 != v20 );
        v29 = v19;
        v30 = v126;
        a1 = v18;
        v31 = v26;
        if ( v29 )
        {
LABEL_63:
          v112 = v26;
          sub_D5F1F0(a4, (__int64)a1);
          v51 = *a1 - 29;
          switch ( *a1 )
          {
            case '*':
            case '+':
            case ',':
            case '-':
            case '.':
            case '/':
            case '0':
            case '1':
            case '2':
            case '3':
            case '4':
            case '5':
            case '6':
            case '7':
            case '8':
            case '9':
            case ':':
            case ';':
              a2 = v51;
              v121 = 257;
              v52 = v30[1];
              v53 = *v30;
              v54 = (unsigned __int8 *)(*(__int64 (__fastcall **)(_QWORD, _QWORD, __int64, __int64))(**(_QWORD **)(a4 + 80) + 16LL))(
                                         *(_QWORD *)(a4 + 80),
                                         v51,
                                         v53,
                                         v52);
              if ( !v54 )
              {
                v125 = 257;
                v54 = (unsigned __int8 *)sub_B504D0(v51, v53, v52, (__int64)&v122, 0, 0);
                if ( (unsigned __int8)sub_920620((__int64)v54) )
                {
                  v98 = *(_QWORD *)(a4 + 96);
                  v99 = *(_DWORD *)(a4 + 104);
                  if ( v98 )
                    sub_B99FD0((__int64)v54, 3u, v98);
                  sub_B45150((__int64)v54, v99);
                }
                a2 = (__int64)v54;
                (*(void (__fastcall **)(_QWORD, unsigned __int8 *, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a4 + 88)
                                                                                            + 16LL))(
                  *(_QWORD *)(a4 + 88),
                  v54,
                  v120,
                  *(_QWORD *)(a4 + 56),
                  *(_QWORD *)(a4 + 64));
                v100 = *(unsigned int **)a4;
                v101 = *(_QWORD *)a4 + 16LL * *(unsigned int *)(a4 + 8);
                if ( *(_QWORD *)a4 != v101 )
                {
                  do
                  {
                    v102 = *((_QWORD *)v100 + 1);
                    a2 = *v100;
                    v100 += 4;
                    sub_B99FD0((__int64)v54, a2, v102);
                  }
                  while ( (unsigned int *)v101 != v100 );
                }
              }
              if ( *v54 <= 0x1Cu )
                goto LABEL_73;
              v55 = *a1;
              if ( (unsigned __int8)v55 <= 0x36u )
              {
                v56 = 0x40540000000000LL;
                if ( _bittest64(&v56, v55) )
                {
                  v57 = sub_B448F0((__int64)a1);
                  sub_B447F0(v54, v57);
                  v58 = sub_B44900((__int64)a1);
                  a2 = v58;
                  sub_B44850(v54, v58);
                  LODWORD(v55) = *a1;
                }
              }
              if ( (unsigned __int8)(v55 - 55) <= 1u || (unsigned int)(v55 - 48) <= 1 )
              {
                a2 = sub_B44E60((__int64)a1);
                sub_B448B0((__int64)v54, a2);
              }
              if ( (unsigned __int8)sub_920620((__int64)a1) )
              {
                a2 = (__int64)a1;
                a1 = v54;
                sub_B45230((__int64)v54, a2);
              }
              else
              {
LABEL_73:
                a1 = v54;
              }
              goto LABEL_33;
            case '?':
              v70 = *v30;
              v71 = v31 - 1;
              v72 = v30 + 1;
              v115 = v112;
              v113 = v70;
              LOBYTE(v70) = a1[1];
              v121 = 257;
              v73 = (unsigned __int8)v70 >> 1;
              v74 = sub_BB5290((__int64)a1);
              v75 = (v73 << 31 >> 31) & 3;
              a2 = v74;
              a1 = (unsigned __int8 *)(*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64 *, __int64, _QWORD))(**(_QWORD **)(a4 + 80) + 64LL))(
                                        *(_QWORD *)(a4 + 80),
                                        v74,
                                        v113,
                                        v72,
                                        v71,
                                        v75);
              if ( a1 )
                goto LABEL_33;
              v125 = 257;
              a1 = (unsigned __int8 *)sub_BD2C40(88, v115);
              if ( !a1 )
                goto LABEL_89;
              v76 = *(_QWORD *)(v113 + 8);
              v77 = v115 & 0x7FFFFFF;
              if ( (unsigned int)*(unsigned __int8 *)(v76 + 8) - 17 <= 1 )
                goto LABEL_88;
              v103 = &v72[v71];
              if ( v72 == v103 )
                goto LABEL_88;
              v104 = v72;
              break;
            case 'C':
            case 'D':
            case 'E':
            case 'F':
            case 'G':
            case 'H':
            case 'I':
            case 'J':
            case 'K':
              v59 = *((_QWORD *)a1 + 1);
              v60 = *(_QWORD *)(*v30 + 8);
              BYTE4(v118) = *(_BYTE *)(v60 + 8) == 18;
              LODWORD(v118) = *(_DWORD *)(v60 + 32);
              if ( (unsigned int)*(unsigned __int8 *)(v59 + 8) - 17 <= 1 )
                v59 = **(_QWORD **)(v59 + 16);
              a2 = v118;
              v61 = sub_BCE1B0((__int64 *)v59, v118);
              v121 = 257;
              v62 = v61;
              v63 = *v30;
              v64 = *a1 - 29;
              if ( v61 == *(_QWORD *)(v63 + 8) )
              {
                a1 = (unsigned __int8 *)v63;
              }
              else
              {
                a2 = v64;
                a1 = (unsigned __int8 *)(*(__int64 (__fastcall **)(_QWORD, _QWORD, __int64, __int64))(**(_QWORD **)(a4 + 80) + 120LL))(
                                          *(_QWORD *)(a4 + 80),
                                          v64,
                                          v63,
                                          v61);
                if ( !a1 )
                {
                  v125 = 257;
                  a1 = (unsigned __int8 *)sub_B51D30(v64, v63, v62, (__int64)&v122, 0, 0);
                  if ( (unsigned __int8)sub_920620((__int64)a1) )
                  {
                    v65 = *(_QWORD *)(a4 + 96);
                    v66 = *(_DWORD *)(a4 + 104);
                    if ( v65 )
                      sub_B99FD0((__int64)a1, 3u, v65);
                    sub_B45150((__int64)a1, v66);
                  }
                  a2 = (__int64)a1;
                  (*(void (__fastcall **)(_QWORD, unsigned __int8 *, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a4 + 88)
                                                                                              + 16LL))(
                    *(_QWORD *)(a4 + 88),
                    a1,
                    v120,
                    *(_QWORD *)(a4 + 56),
                    *(_QWORD *)(a4 + 64));
                  v67 = *(unsigned int **)a4;
                  v68 = *(_QWORD *)a4 + 16LL * *(unsigned int *)(a4 + 8);
                  if ( *(_QWORD *)a4 != v68 )
                  {
                    do
                    {
                      v69 = *((_QWORD *)v67 + 1);
                      a2 = *v67;
                      v67 += 4;
                      sub_B99FD0((__int64)a1, a2, v69);
                    }
                    while ( (unsigned int *)v68 != v67 );
                  }
                }
              }
              goto LABEL_33;
            case 'R':
              v121 = 257;
              v84 = v30[1];
              v85 = *v30;
              v86 = *v30;
              v87 = *((_WORD *)a1 + 1) & 0x3F;
              a2 = *((_WORD *)a1 + 1) & 0x3F;
              a1 = (unsigned __int8 *)(*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(a4 + 80) + 56LL))(
                                        *(_QWORD *)(a4 + 80),
                                        a2,
                                        v86,
                                        v84);
              if ( !a1 )
              {
                v125 = 257;
                a1 = (unsigned __int8 *)sub_BD2C40(72, unk_3F10FD0);
                if ( a1 )
                {
                  v88 = *(_QWORD ***)(v85 + 8);
                  v89 = *((unsigned __int8 *)v88 + 8);
                  if ( (unsigned int)(v89 - 17) > 1 )
                  {
                    v91 = sub_BCB2A0(*v88);
                  }
                  else
                  {
                    BYTE4(v119) = (_BYTE)v89 == 18;
                    LODWORD(v119) = *((_DWORD *)v88 + 8);
                    v90 = (__int64 *)sub_BCB2A0(*v88);
                    v91 = sub_BCE1B0(v90, v119);
                  }
                  sub_B523C0((__int64)a1, v91, 53, v87, v85, v84, (__int64)&v122, 0, 0, 0);
                }
                a2 = (__int64)a1;
                (*(void (__fastcall **)(_QWORD, unsigned __int8 *, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a4 + 88)
                                                                                            + 16LL))(
                  *(_QWORD *)(a4 + 88),
                  a1,
                  v120,
                  *(_QWORD *)(a4 + 56),
                  *(_QWORD *)(a4 + 64));
                v92 = *(unsigned int **)a4;
                v93 = *(_QWORD *)a4 + 16LL * *(unsigned int *)(a4 + 8);
                if ( *(_QWORD *)a4 != v93 )
                {
                  do
                  {
                    v94 = *((_QWORD *)v92 + 1);
                    a2 = *v92;
                    v92 += 4;
                    sub_B99FD0((__int64)a1, a2, v94);
                  }
                  while ( (unsigned int *)v93 != v92 );
                }
              }
              goto LABEL_33;
            case 'S':
              v125 = 257;
              v81 = *((_WORD *)a1 + 1);
              v82 = v30[1];
              v83 = *v30;
              HIDWORD(v120[0]) = 0;
              a2 = v81 & 0x3F;
              a1 = (unsigned __int8 *)sub_B35C90(a4, a2, v83, v82, (__int64)&v122, 0, LODWORD(v120[0]), 0);
              goto LABEL_33;
            default:
              goto LABEL_124;
          }
          while ( 1 )
          {
            v105 = *(_QWORD *)(*v104 + 8);
            v106 = *(unsigned __int8 *)(v105 + 8);
            if ( v106 == 17 )
              break;
            if ( v106 == 18 )
            {
              v107 = 18;
              goto LABEL_122;
            }
            if ( v103 == ++v104 )
              goto LABEL_88;
          }
          v107 = 17;
LABEL_122:
          BYTE4(v119) = v107 == 18;
          LODWORD(v119) = *(_DWORD *)(v105 + 32);
          v108 = sub_BCE1B0(*(__int64 **)(v113 + 8), v119);
          v77 = v115 & 0x7FFFFFF;
          v76 = v108;
LABEL_88:
          sub_B44260((__int64)a1, v76, 34, v77, 0, 0);
          *((_QWORD *)a1 + 9) = v74;
          *((_QWORD *)a1 + 10) = sub_B4DC50(v74, (__int64)v72, v71);
          sub_B4D9A0((__int64)a1, v113, v72, v71, (__int64)&v122);
LABEL_89:
          sub_B4DDE0((__int64)a1, v75);
          a2 = (__int64)a1;
          (*(void (__fastcall **)(_QWORD, unsigned __int8 *, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a4 + 88) + 16LL))(
            *(_QWORD *)(a4 + 88),
            a1,
            v120,
            *(_QWORD *)(a4 + 56),
            *(_QWORD *)(a4 + 64));
          v78 = *(unsigned int **)a4;
          v79 = *(_QWORD *)a4 + 16LL * *(unsigned int *)(a4 + 8);
          if ( *(_QWORD *)a4 != v79 )
          {
            do
            {
              v80 = *((_QWORD *)v78 + 1);
              a2 = *v78;
              v78 += 4;
              sub_B99FD0((__int64)a1, a2, v80);
            }
            while ( (unsigned int *)v79 != v78 );
          }
        }
LABEL_33:
        if ( v126 != &v128 )
          _libc_free(v126, a2);
        return (__int64)a1;
      case '[':
        if ( (a1[7] & 0x40) != 0 )
          a2 = *((_QWORD *)a1 - 1);
        else
          a2 = (__int64)&a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
        v32 = *(_QWORD *)(a2 + 64);
        v33 = *(_DWORD *)(v32 + 32);
        if ( v33 > 0x40 )
        {
          v50 = sub_C444A0(v32 + 24);
          v34 = -1;
          if ( v33 - v50 <= 0x40 )
            v34 = **(_DWORD **)(v32 + 24);
        }
        else
        {
          v34 = *(_DWORD *)(v32 + 24);
        }
        if ( !(_DWORD)a3 )
          goto LABEL_56;
        v35 = 0;
        while ( 2 )
        {
          if ( v8[v35] == v34 )
          {
            v36 = sub_11B0C30(*(_QWORD *)a2, v8, a3, a4);
            sub_D5F1F0(a4, (__int64)a1);
            v125 = 257;
            if ( (a1[7] & 0x40) != 0 )
              v37 = (unsigned __int8 *)*((_QWORD *)a1 - 1);
            else
              v37 = &a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
            v38 = *((_QWORD *)v37 + 4);
            v39 = sub_BCB2E0(*(_QWORD **)(a4 + 72));
            v40 = sub_ACD640(v39, v35, 0);
            result = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(a4 + 80) + 104LL))(
                       *(_QWORD *)(a4 + 80),
                       v36,
                       v38,
                       v40);
            if ( !result )
            {
              v131 = 257;
              v41 = sub_BD2C40(72, 3u);
              if ( v41 )
              {
                v111 = v41;
                sub_B4DFA0((__int64)v41, v36, v38, v40, (__int64)&v126, v42, 0, 0);
                v41 = v111;
              }
              v43 = a4;
              v117 = (__int64)v41;
              (*(void (__fastcall **)(_QWORD, _QWORD *, _BYTE **, _QWORD, _QWORD))(**(_QWORD **)(v43 + 88) + 16LL))(
                *(_QWORD *)(v43 + 88),
                v41,
                &v122,
                *(_QWORD *)(v43 + 56),
                *(_QWORD *)(v43 + 64));
              v44 = v43;
              v45 = *(_QWORD *)v43;
              v46 = *(unsigned int *)(v44 + 8);
              result = v117;
              v47 = v45 + 16 * v46;
              if ( v45 != v47 )
              {
                do
                {
                  v48 = *(_QWORD *)(v45 + 8);
                  v49 = *(_DWORD *)v45;
                  v45 += 16;
                  sub_B99FD0(v117, v49, v48);
                }
                while ( v47 != v45 );
                return v117;
              }
            }
            return result;
          }
          if ( (_DWORD)a3 - 1 != v35 )
          {
            ++v35;
            continue;
          }
          break;
        }
LABEL_56:
        a1 = *(unsigned __int8 **)a2;
        continue;
      default:
LABEL_124:
        BUG();
    }
  }
}
