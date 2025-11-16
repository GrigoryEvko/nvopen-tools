// Function: sub_2DC8FC0
// Address: 0x2dc8fc0
//
__int64 __fastcall sub_2DC8FC0(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // rbx
  __int64 i; // r12
  __int64 v5; // rax
  _BYTE **v6; // r14
  _BYTE *v7; // r12
  __int64 v8; // rbx
  int v9; // edx
  unsigned int v10; // ecx
  unsigned __int8 v11; // al
  int v12; // ebx
  __int64 v13; // rax
  int v14; // r13d
  _QWORD *v15; // rax
  __int64 v16; // rax
  __int64 v17; // rsi
  __int64 v18; // r9
  __int64 v19; // r8
  unsigned int *v20; // rax
  int v21; // ecx
  unsigned int *v22; // rdx
  int v23; // eax
  __int16 v24; // kr00_2
  unsigned __int64 v25; // rbx
  int v26; // eax
  int v27; // eax
  __int64 v28; // r9
  __int64 v29; // r9
  int v31; // eax
  int v32; // eax
  unsigned int v33; // r13d
  int v34; // eax
  unsigned __int8 *v35; // rbx
  __int64 (__fastcall *v36)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *); // rax
  char v37; // al
  __int64 v38; // r9
  int v39; // r13d
  unsigned int *v40; // r13
  unsigned int *v41; // r12
  __int64 v42; // rbx
  __int64 v43; // rdx
  unsigned int v44; // esi
  __int64 v45; // rax
  __int64 v46; // rsi
  __int64 v47; // r11
  __int64 (__fastcall *v48)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v49; // rax
  __int64 v50; // r9
  __int64 v51; // rdi
  _BYTE *v52; // rax
  __int64 v53; // rax
  __int64 v54; // r8
  __int64 v55; // r9
  __int64 v56; // rax
  unsigned __int64 v57; // rdx
  int v58; // eax
  __int64 v59; // r9
  unsigned __int64 v60; // rsi
  unsigned __int64 v61; // rax
  __int64 *v62; // rax
  _BYTE *v63; // rax
  char v64; // al
  __int64 v65; // r9
  int v66; // ebx
  __int64 v67; // rbx
  unsigned int *v68; // rbx
  __int64 v69; // r12
  __int64 v70; // rdx
  __int64 v71; // rax
  __int64 v72; // [rsp+0h] [rbp-1C0h]
  int v73; // [rsp+14h] [rbp-1ACh]
  unsigned __int8 *v74; // [rsp+18h] [rbp-1A8h]
  unsigned int v76; // [rsp+28h] [rbp-198h]
  __int64 v77; // [rsp+30h] [rbp-190h]
  __int64 v78; // [rsp+30h] [rbp-190h]
  __int64 v79; // [rsp+30h] [rbp-190h]
  _BYTE *v80; // [rsp+30h] [rbp-190h]
  unsigned __int8 v81; // [rsp+38h] [rbp-188h]
  __int64 v82; // [rsp+38h] [rbp-188h]
  __int64 v83; // [rsp+38h] [rbp-188h]
  __int64 v84; // [rsp+38h] [rbp-188h]
  _BYTE *v85; // [rsp+38h] [rbp-188h]
  __int64 v86; // [rsp+38h] [rbp-188h]
  __int64 **v87; // [rsp+38h] [rbp-188h]
  __int64 v88; // [rsp+38h] [rbp-188h]
  __int64 v89; // [rsp+38h] [rbp-188h]
  __int64 v90; // [rsp+38h] [rbp-188h]
  unsigned int *v91; // [rsp+38h] [rbp-188h]
  __int64 v92; // [rsp+38h] [rbp-188h]
  _BYTE *v93; // [rsp+50h] [rbp-170h]
  int v94; // [rsp+58h] [rbp-168h]
  char v95; // [rsp+5Fh] [rbp-161h]
  __int64 v96; // [rsp+60h] [rbp-160h]
  unsigned __int64 v97; // [rsp+60h] [rbp-160h]
  __int64 v98; // [rsp+68h] [rbp-158h]
  _QWORD v99[4]; // [rsp+70h] [rbp-150h] BYREF
  __int16 v100; // [rsp+90h] [rbp-130h]
  __int64 v101[4]; // [rsp+A0h] [rbp-120h] BYREF
  __int16 v102; // [rsp+C0h] [rbp-100h]
  _BYTE *v103; // [rsp+D0h] [rbp-F0h] BYREF
  __int64 v104; // [rsp+D8h] [rbp-E8h]
  _BYTE v105[32]; // [rsp+E0h] [rbp-E0h] BYREF
  unsigned int *v106; // [rsp+100h] [rbp-C0h] BYREF
  __int64 v107; // [rsp+108h] [rbp-B8h]
  _BYTE v108[32]; // [rsp+110h] [rbp-B0h] BYREF
  __int64 v109; // [rsp+130h] [rbp-90h]
  _BYTE *v110; // [rsp+138h] [rbp-88h]
  __int64 v111; // [rsp+140h] [rbp-80h]
  _QWORD *v112; // [rsp+148h] [rbp-78h]
  void **v113; // [rsp+150h] [rbp-70h]
  void **v114; // [rsp+158h] [rbp-68h]
  __int64 v115; // [rsp+160h] [rbp-60h]
  int v116; // [rsp+168h] [rbp-58h]
  __int16 v117; // [rsp+16Ch] [rbp-54h]
  char v118; // [rsp+16Eh] [rbp-52h]
  __int64 v119; // [rsp+170h] [rbp-50h]
  __int64 v120; // [rsp+178h] [rbp-48h]
  void *v121; // [rsp+180h] [rbp-40h] BYREF
  void *v122; // [rsp+188h] [rbp-38h] BYREF

  v2 = a1 + 72;
  v3 = *(_QWORD *)(a1 + 80);
  v103 = v105;
  v104 = 0x400000000LL;
  if ( a1 + 72 == v3 )
  {
    i = 0;
  }
  else
  {
    if ( !v3 )
      BUG();
    while ( 1 )
    {
      i = *(_QWORD *)(v3 + 32);
      if ( i != v3 + 24 )
        break;
      v3 = *(_QWORD *)(v3 + 8);
      if ( v2 == v3 )
        break;
      if ( !v3 )
        BUG();
    }
  }
  while ( v3 != v2 )
  {
    if ( !i )
      BUG();
    if ( *(_BYTE *)(i - 24) == 85 )
    {
      v53 = *(_QWORD *)(i - 56);
      if ( v53 )
      {
        if ( !*(_BYTE *)v53
          && *(_QWORD *)(v53 + 24) == *(_QWORD *)(i + 56)
          && (*(_BYTE *)(v53 + 33) & 0x20) != 0
          && (unsigned int)(*(_DWORD *)(v53 + 36) - 387) <= 0xE
          && ((1LL << (*(_BYTE *)(v53 + 36) + 125)) & 0x7FAF) != 0
          && (unsigned __int8)sub_DFE550(a2) )
        {
          v56 = (unsigned int)v104;
          v57 = (unsigned int)v104 + 1LL;
          if ( v57 > HIDWORD(v104) )
          {
            sub_C8D5F0((__int64)&v103, v105, v57, 8u, v54, v55);
            v56 = (unsigned int)v104;
          }
          *(_QWORD *)&v103[8 * v56] = i - 24;
          LODWORD(v104) = v104 + 1;
        }
      }
    }
    for ( i = *(_QWORD *)(i + 8); ; i = *(_QWORD *)(v3 + 32) )
    {
      v5 = v3 - 24;
      if ( !v3 )
        v5 = 0;
      if ( i != v5 + 48 )
        break;
      v3 = *(_QWORD *)(v3 + 8);
      if ( v2 == v3 )
        break;
      if ( !v3 )
        BUG();
    }
  }
  v93 = &v103[8 * (unsigned int)v104];
  if ( v103 != v93 )
  {
    v81 = 0;
    v6 = (_BYTE **)v103;
    while ( 1 )
    {
      v7 = *v6;
      if ( **v6 > 0x1Cu )
      {
        switch ( **v6 )
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
            goto LABEL_114;
          case 'T':
          case 'U':
          case 'V':
            v8 = *((_QWORD *)v7 + 1);
            v9 = *(unsigned __int8 *)(v8 + 8);
            v10 = v9 - 17;
            v11 = *(_BYTE *)(v8 + 8);
            if ( (unsigned int)(v9 - 17) <= 1 )
              v11 = *(_BYTE *)(**(_QWORD **)(v8 + 16) + 8LL);
            if ( v11 <= 3u || v11 == 5 || (v11 & 0xFD) == 4 )
              goto LABEL_114;
            if ( (_BYTE)v9 == 15 )
            {
              if ( (*(_BYTE *)(v8 + 9) & 4) == 0 || !sub_BCB420(*((_QWORD *)v7 + 1)) )
                break;
              v62 = *(__int64 **)(v8 + 16);
              v8 = *v62;
              v9 = *(unsigned __int8 *)(*v62 + 8);
              v10 = v9 - 17;
            }
            else if ( (_BYTE)v9 == 16 )
            {
              do
              {
                v8 = *(_QWORD *)(v8 + 24);
                LOBYTE(v9) = *(_BYTE *)(v8 + 8);
              }
              while ( (_BYTE)v9 == 16 );
              v10 = (unsigned __int8)v9 - 17;
            }
            if ( v10 <= 1 )
              LOBYTE(v9) = *(_BYTE *)(**(_QWORD **)(v8 + 16) + 8LL);
            if ( (unsigned __int8)v9 <= 3u || (_BYTE)v9 == 5 || (v9 & 0xFD) == 4 )
            {
LABEL_114:
              v12 = sub_B45210((__int64)v7);
              goto LABEL_29;
            }
            break;
          default:
            break;
        }
      }
      v12 = 0;
LABEL_29:
      v13 = *((_QWORD *)v7 - 4);
      if ( !v13 || *(_BYTE *)v13 || *(_QWORD *)(v13 + 24) != *((_QWORD *)v7 + 10) )
        BUG();
      v14 = *(_DWORD *)(v13 + 36);
      v76 = sub_F6F0E0(v14);
      v73 = sub_DFE580(a2);
      v15 = (_QWORD *)sub_BD5C60((__int64)v7);
      v109 = 0;
      v112 = v15;
      v113 = &v121;
      v110 = 0;
      v114 = &v122;
      v121 = &unk_49DA100;
      v115 = 0;
      v116 = 0;
      v117 = 512;
      v118 = 7;
      v119 = 0;
      v120 = 0;
      LOWORD(v111) = 0;
      v122 = &unk_49DA0B0;
      v16 = *((_QWORD *)v7 + 5);
      v107 = 0x200000000LL;
      v109 = v16;
      v106 = (unsigned int *)v108;
      v110 = v7 + 24;
      v17 = *(_QWORD *)sub_B46C60((__int64)v7);
      v101[0] = v17;
      if ( !v17 )
        break;
      sub_B96E90((__int64)v101, v17, 1);
      v19 = v101[0];
      if ( !v101[0] )
        break;
      v20 = v106;
      v21 = v107;
      v22 = &v106[4 * (unsigned int)v107];
      if ( v106 == v22 )
      {
LABEL_93:
        if ( (unsigned int)v107 >= (unsigned __int64)HIDWORD(v107) )
        {
          v60 = (unsigned int)v107 + 1LL;
          v61 = v72 & 0xFFFFFFFF00000000LL;
          v72 &= 0xFFFFFFFF00000000LL;
          if ( HIDWORD(v107) < v60 )
          {
            v97 = v61;
            v98 = v101[0];
            sub_C8D5F0((__int64)&v106, v108, v60, 0x10u, v101[0], v18);
            v61 = v97;
            v19 = v98;
            v22 = &v106[4 * (unsigned int)v107];
          }
          *(_QWORD *)v22 = v61;
          *((_QWORD *)v22 + 1) = v19;
          v19 = v101[0];
          LODWORD(v107) = v107 + 1;
        }
        else
        {
          if ( v22 )
          {
            *v22 = 0;
            *((_QWORD *)v22 + 1) = v19;
            v21 = v107;
            v19 = v101[0];
          }
          LODWORD(v107) = v21 + 1;
        }
LABEL_97:
        if ( !v19 )
          goto LABEL_40;
        goto LABEL_39;
      }
      while ( *v20 )
      {
        v20 += 4;
        if ( v22 == v20 )
          goto LABEL_93;
      }
      *((_QWORD *)v20 + 1) = v101[0];
LABEL_39:
      sub_B91220((__int64)v101, v19);
LABEL_40:
      v23 = v116;
      v116 = v12;
      v94 = v23;
      v96 = v115;
      v24 = v117;
      v95 = v118;
      switch ( v14 )
      {
        case 387:
        case 395:
        case 397:
        case 398:
        case 399:
        case 400:
        case 401:
          v25 = *(_QWORD *)&v7[-32 * (*((_DWORD *)v7 + 1) & 0x7FFFFFF)];
          v26 = *(_DWORD *)(*(_QWORD *)(v25 + 8) + 32LL);
          if ( v26 && (v26 & (v26 - 1)) == 0 )
            goto LABEL_43;
          goto LABEL_62;
        case 388:
        case 396:
          v25 = *(_QWORD *)&v7[-32 * (*((_DWORD *)v7 + 1) & 0x7FFFFFF)];
          v45 = *(_QWORD *)(v25 + 8);
          LODWORD(v46) = *(_DWORD *)(v45 + 32);
          if ( !(_DWORD)v46 || ((unsigned int)v46 & ((_DWORD)v46 - 1)) != 0 )
            goto LABEL_62;
          v86 = *(_QWORD *)(v45 + 24);
          v46 = (unsigned int)v46;
          if ( v86 != sub_BCB2A0(v112) )
          {
LABEL_43:
            v27 = sub_F6EED0(v14);
            v29 = sub_F6F7A0((__int64 *)&v106, (void **)v25, v27, v73, v76, v28);
            goto LABEL_44;
          }
          v100 = 257;
          v47 = sub_BCD140(v112, v46);
          if ( v47 == *(_QWORD *)(v25 + 8) )
          {
            v50 = v25;
            goto LABEL_91;
          }
          v48 = (__int64 (__fastcall *)(__int64, unsigned int, _BYTE *, __int64))*((_QWORD *)*v113 + 15);
          if ( v48 == sub_920130 )
          {
            if ( *(_BYTE *)v25 > 0x15u )
              goto LABEL_126;
            v87 = (__int64 **)v47;
            v46 = v25;
            if ( (unsigned __int8)sub_AC4810(0x31u) )
              v49 = sub_ADAB70(49, v25, v87, 0);
            else
              v49 = sub_AA93C0(0x31u, v25, (__int64)v87);
            v47 = (__int64)v87;
            v50 = v49;
          }
          else
          {
            v92 = v47;
            v46 = 49;
            v71 = v48((__int64)v113, 49u, (_BYTE *)v25, v47);
            v47 = v92;
            v50 = v71;
          }
          if ( v50 )
            goto LABEL_91;
LABEL_126:
          v102 = 257;
          v89 = sub_B51D30(49, v25, v47, (__int64)v101, 0, 0);
          v64 = sub_920620(v89);
          v65 = v89;
          if ( v64 )
          {
            v66 = v116;
            if ( v115 )
            {
              sub_B99FD0(v89, 3u, v115);
              v65 = v89;
            }
            v90 = v65;
            sub_B45150(v65, v66);
            v65 = v90;
          }
          v46 = v65;
          v79 = v65;
          (*((void (__fastcall **)(void **, __int64, _QWORD *, _BYTE *, __int64))*v114 + 2))(v114, v65, v99, v110, v111);
          v50 = v79;
          v67 = 4LL * (unsigned int)v107;
          v91 = &v106[v67];
          if ( v106 != &v106[v67] )
          {
            v80 = v7;
            v68 = v106;
            v69 = v50;
            do
            {
              v70 = *((_QWORD *)v68 + 1);
              v46 = *v68;
              v68 += 4;
              sub_B99FD0(v69, v46, v70);
            }
            while ( v91 != v68 );
            v50 = v69;
            v7 = v80;
          }
LABEL_91:
          v102 = 257;
          v51 = *(_QWORD *)(v50 + 8);
          v88 = v50;
          if ( v14 == 388 )
          {
            v63 = (_BYTE *)sub_AD62B0(v51);
            v29 = sub_92B530(&v106, 0x20u, v88, v63, (__int64)v101);
          }
          else
          {
            v52 = (_BYTE *)sub_AD6530(v51, v46);
            v29 = sub_92B530(&v106, 0x21u, v88, v52, (__int64)v101);
          }
          goto LABEL_44;
        case 389:
        case 394:
          v74 = *(unsigned __int8 **)&v7[-32 * (*((_DWORD *)v7 + 1) & 0x7FFFFFF)];
          v78 = *(_QWORD *)&v7[32 * (1LL - (*((_DWORD *)v7 + 1) & 0x7FFFFFF))];
          v32 = sub_F6EED0(v14);
          v33 = v32;
          if ( (v12 & 1) == 0 )
          {
            v29 = sub_F6F5C0((__int64)&v106, (__int64)v74, v78, v32, v76);
            goto LABEL_44;
          }
          v34 = *(_DWORD *)(*(_QWORD *)(v78 + 8) + 32LL);
          if ( v34 && (v34 & (v34 - 1)) == 0 )
          {
            v35 = (unsigned __int8 *)sub_F6F7A0((__int64 *)&v106, (void **)v78, v33, v73, v76, v78);
            v100 = 259;
            v99[0] = "bin.rdx";
            v36 = (__int64 (__fastcall *)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *))*((_QWORD *)*v113 + 2);
            if ( v36 != sub_9202E0 )
            {
              v29 = v36((__int64)v113, v33, v74, v35);
              goto LABEL_72;
            }
            if ( *v74 > 0x15u || *v35 > 0x15u )
              goto LABEL_73;
            v29 = (unsigned __int8)sub_AC47B0(v33)
                ? sub_AD5570(v33, (__int64)v74, v35, 0, 0)
                : sub_AABE40(v33, v74, v35);
LABEL_72:
            if ( !v29 )
            {
LABEL_73:
              v102 = 257;
              v82 = sub_B504D0(v33, (__int64)v74, (__int64)v35, (__int64)v101, 0, 0);
              v37 = sub_920620(v82);
              v38 = v82;
              if ( v37 )
              {
                v39 = v116;
                if ( v115 )
                {
                  sub_B99FD0(v82, 3u, v115);
                  v38 = v82;
                }
                v83 = v38;
                sub_B45150(v38, v39);
                v38 = v83;
              }
              v84 = v38;
              (*((void (__fastcall **)(void **, __int64, _QWORD *, _BYTE *, __int64))*v114 + 2))(
                v114,
                v38,
                v99,
                v110,
                v111);
              v40 = v106;
              v29 = v84;
              if ( v106 != &v106[4 * (unsigned int)v107] )
              {
                v85 = v7;
                v41 = &v106[4 * (unsigned int)v107];
                v42 = v29;
                do
                {
                  v43 = *((_QWORD *)v40 + 1);
                  v44 = *v40;
                  v40 += 4;
                  sub_B99FD0(v42, v44, v43);
                }
                while ( v41 != v40 );
                v7 = v85;
                v29 = v42;
              }
            }
LABEL_44:
            sub_BD84D0((__int64)v7, v29);
            sub_B43D60(v7);
            v116 = v94;
            v115 = v96;
            v117 = v24;
            v118 = v95;
            nullsub_61();
            v121 = &unk_49DA100;
            nullsub_63();
            if ( v106 != (unsigned int *)v108 )
              _libc_free((unsigned __int64)v106);
            v81 = 1;
            goto LABEL_47;
          }
LABEL_62:
          v116 = v94;
          v115 = v96;
          v117 = v24;
          v118 = v95;
          nullsub_61();
          v121 = &unk_49DA100;
          nullsub_63();
          if ( v106 != (unsigned int *)v108 )
            _libc_free((unsigned __int64)v106);
LABEL_47:
          if ( v93 == (_BYTE *)++v6 )
          {
            v93 = v103;
            goto LABEL_49;
          }
          break;
        case 390:
        case 392:
          v77 = *(_QWORD *)&v7[-32 * (*((_DWORD *)v7 + 1) & 0x7FFFFFF)];
          v31 = *(_DWORD *)(*(_QWORD *)(v77 + 8) + 32LL);
          if ( !v31 || (v31 & (v31 - 1)) != 0 || (v12 & 2) == 0 )
            goto LABEL_62;
          v58 = sub_F6EED0(v14);
          v29 = sub_F6F7A0((__int64 *)&v106, (void **)v77, v58, v73, v76, v59);
          goto LABEL_44;
        default:
          BUG();
      }
    }
    sub_93FB40((__int64)&v106, 0);
    v19 = v101[0];
    goto LABEL_97;
  }
  v81 = 0;
LABEL_49:
  if ( v93 != v105 )
    _libc_free((unsigned __int64)v93);
  return v81;
}
