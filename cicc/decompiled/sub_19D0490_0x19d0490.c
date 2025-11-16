// Function: sub_19D0490
// Address: 0x19d0490
//
__int64 __fastcall sub_19D0490(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rax
  _QWORD *v7; // rbx
  _QWORD *v8; // r12
  int v9; // edx
  __int64 v10; // r15
  __int64 v11; // rsi
  __int64 v12; // rdx
  __int64 v13; // r9
  __int64 v14; // r10
  __int64 v15; // rsi
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rdi
  int v19; // eax
  __int64 v20; // rax
  __int64 v21; // rax
  _QWORD *v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rax
  _QWORD *v25; // r14
  int v26; // r15d
  __int64 v27; // rax
  _QWORD *v28; // rax
  __int64 v29; // rax
  unsigned __int8 *v30; // rsi
  __int64 v31; // r12
  _BYTE *v32; // r14
  unsigned int v33; // eax
  unsigned __int64 v34; // rdi
  __int64 v35; // rax
  __int64 v36; // rdx
  __int64 v37; // rsi
  unsigned int v38; // ebx
  _QWORD *v39; // r15
  __int64 v40; // rax
  __int64 *v41; // rax
  __int64 v42; // r15
  unsigned __int8 **v43; // rbx
  unsigned __int8 *v44; // rsi
  _QWORD **v45; // rdx
  _QWORD **v46; // rbx
  _QWORD **v47; // r15
  _QWORD *v48; // r12
  _BYTE *v49; // r12
  _BYTE *v50; // rbx
  unsigned __int64 v51; // rdi
  __int64 result; // rax
  unsigned int v53; // ebx
  unsigned int v54; // eax
  __int64 v55; // rsi
  unsigned __int8 *v56; // rsi
  _QWORD *v57; // rax
  unsigned int v58; // eax
  __int64 v59; // rdi
  __int64 v60; // rcx
  __int64 v61; // rsi
  unsigned __int64 v62; // r8
  unsigned int v63; // esi
  int v64; // eax
  __int64 v65; // rax
  int v66; // eax
  __int64 v67; // rdx
  unsigned __int64 v68; // rax
  __int64 v69; // rax
  __int64 v70; // rdi
  __int64 v71; // rsi
  unsigned __int64 v72; // r14
  int v73; // eax
  __int64 v74; // rax
  __int64 v75; // rax
  __int64 v76; // rax
  __int64 v77; // rax
  __int64 v78; // [rsp+8h] [rbp-698h]
  __int64 v79; // [rsp+10h] [rbp-690h]
  unsigned __int64 v80; // [rsp+18h] [rbp-688h]
  __int64 v81; // [rsp+20h] [rbp-680h]
  __int64 v82; // [rsp+20h] [rbp-680h]
  __int64 v83; // [rsp+28h] [rbp-678h]
  __int64 v84; // [rsp+28h] [rbp-678h]
  __int64 v85; // [rsp+28h] [rbp-678h]
  __int64 v86; // [rsp+30h] [rbp-670h]
  __int64 v87; // [rsp+30h] [rbp-670h]
  __int64 v88; // [rsp+30h] [rbp-670h]
  unsigned __int64 v89; // [rsp+30h] [rbp-670h]
  __int64 v90; // [rsp+40h] [rbp-660h]
  __int64 v91; // [rsp+40h] [rbp-660h]
  __int64 v92; // [rsp+40h] [rbp-660h]
  unsigned __int64 v93; // [rsp+40h] [rbp-660h]
  unsigned __int64 v94; // [rsp+40h] [rbp-660h]
  __int64 v95; // [rsp+40h] [rbp-660h]
  int v96; // [rsp+48h] [rbp-658h]
  __int64 v97; // [rsp+48h] [rbp-658h]
  __int64 v98; // [rsp+48h] [rbp-658h]
  __int64 v99; // [rsp+48h] [rbp-658h]
  __int64 v100; // [rsp+48h] [rbp-658h]
  __int64 v101; // [rsp+48h] [rbp-658h]
  __int64 v102; // [rsp+48h] [rbp-658h]
  __int64 v103; // [rsp+48h] [rbp-658h]
  __int64 v104; // [rsp+48h] [rbp-658h]
  __int64 v105; // [rsp+50h] [rbp-650h]
  __int64 v106; // [rsp+58h] [rbp-648h]
  __int64 v107; // [rsp+60h] [rbp-640h]
  _BYTE *v108; // [rsp+68h] [rbp-638h]
  unsigned __int8 *v109; // [rsp+78h] [rbp-628h] BYREF
  __int64 v110[3]; // [rsp+80h] [rbp-620h] BYREF
  _QWORD *v111; // [rsp+98h] [rbp-608h]
  __int64 v112; // [rsp+A0h] [rbp-600h]
  int v113; // [rsp+A8h] [rbp-5F8h]
  __int64 v114; // [rsp+B0h] [rbp-5F0h]
  __int64 v115; // [rsp+B8h] [rbp-5E8h]
  _BYTE *v116; // [rsp+D0h] [rbp-5D0h] BYREF
  __int64 v117; // [rsp+D8h] [rbp-5C8h]
  _BYTE v118[672]; // [rsp+E0h] [rbp-5C0h] BYREF
  int v119; // [rsp+380h] [rbp-320h]
  __int64 v120; // [rsp+660h] [rbp-40h]

  v107 = a4;
  v108 = (_BYTE *)a2;
  v6 = sub_15F2050(a2);
  v116 = v118;
  v106 = sub_1632FA0(v6);
  v117 = 0x800000000LL;
  v120 = v106;
  if ( !a2 )
    BUG();
  v7 = *(_QWORD **)(a2 + 32);
  while ( 1 )
  {
    if ( !v7 )
      BUG();
    v8 = v7 - 3;
    v9 = *((unsigned __int8 *)v7 - 8);
    if ( (unsigned int)(v9 - 25) <= 9 )
      break;
    if ( (_BYTE)v9 == 55 )
    {
      if ( sub_15F32D0((__int64)(v7 - 3))
        || (*((_BYTE *)v7 - 6) & 1) != 0
        || v107 != sub_14ABE30((unsigned __int8 *)*(v7 - 9))
        || !(unsigned __int8)sub_19CFE90(a3, *(v7 - 6), v110, v106) )
      {
        break;
      }
      v13 = 1;
      v14 = v110[0];
      v15 = *(_QWORD *)*(v7 - 9);
      while ( 2 )
      {
        switch ( *(_BYTE *)(v15 + 8) )
        {
          case 0:
          case 8:
          case 0xA:
          case 0xC:
          case 0x10:
            v65 = *(_QWORD *)(v15 + 32);
            v15 = *(_QWORD *)(v15 + 24);
            v13 *= v65;
            continue;
          case 1:
            v12 = 16;
            break;
          case 2:
            v12 = 32;
            break;
          case 3:
          case 9:
            v12 = 64;
            break;
          case 4:
            v12 = 80;
            break;
          case 5:
          case 6:
            v12 = 128;
            break;
          case 7:
            v101 = v110[0];
            v105 = v13;
            v66 = sub_15A9520(v120, 0);
            v13 = v105;
            v14 = v101;
            v12 = (unsigned int)(8 * v66);
            break;
          case 0xB:
            v12 = *(_DWORD *)(v15 + 8) >> 8;
            break;
          case 0xD:
            v98 = v110[0];
            v105 = v13;
            v57 = (_QWORD *)sub_15A9930(v120, v15);
            v13 = v105;
            v14 = v98;
            v12 = 8LL * *v57;
            break;
          case 0xE:
            v83 = v110[0];
            v86 = v13;
            v91 = *(_QWORD *)(v15 + 24);
            v99 = v120;
            v105 = *(_QWORD *)(v15 + 32);
            v58 = sub_15A9FE0(v120, v91);
            v14 = v83;
            v59 = v99;
            v60 = 1;
            v61 = v91;
            v13 = v86;
            v62 = v58;
            while ( 2 )
            {
              switch ( *(_BYTE *)(v61 + 8) )
              {
                case 0:
                case 8:
                case 0xA:
                case 0xC:
                case 0x10:
                  v74 = *(_QWORD *)(v61 + 32);
                  v61 = *(_QWORD *)(v61 + 24);
                  v60 *= v74;
                  continue;
                case 1:
                  v68 = 16;
                  goto LABEL_103;
                case 2:
                  v68 = 32;
                  goto LABEL_103;
                case 3:
                case 9:
                  v68 = 64;
                  goto LABEL_103;
                case 4:
                  v68 = 80;
                  goto LABEL_103;
                case 5:
                case 6:
                  v68 = 128;
                  goto LABEL_103;
                case 7:
                  v94 = v62;
                  v104 = v60;
                  v73 = sub_15A9520(v59, 0);
                  v60 = v104;
                  v62 = v94;
                  v13 = v86;
                  v14 = v83;
                  v68 = (unsigned int)(8 * v73);
                  goto LABEL_103;
                case 0xB:
                  v68 = *(_DWORD *)(v61 + 8) >> 8;
                  goto LABEL_103;
                case 0xD:
                  sub_15A9930(v99, v61);
                  JUMPOUT(0x19D10C7);
                case 0xE:
                  v78 = v83;
                  v79 = v86;
                  v80 = v62;
                  v81 = v60;
                  v84 = *(_QWORD *)(v61 + 24);
                  v88 = v99;
                  v103 = *(_QWORD *)(v61 + 32);
                  v93 = (unsigned int)sub_15A9FE0(v59, v84);
                  v75 = sub_127FA20(v88, v84);
                  v60 = v81;
                  v62 = v80;
                  v13 = v79;
                  v14 = v78;
                  v68 = 8 * v103 * v93 * ((v93 + ((unsigned __int64)(v75 + 7) >> 3) - 1) / v93);
LABEL_103:
                  v12 = 8 * v62 * v105 * ((v62 + ((v68 * v60 + 7) >> 3) - 1) / v62);
                  break;
                case 0xF:
                  JUMPOUT(0x19D0F8D);
              }
              return result;
            }
          case 0xF:
            v63 = *(_DWORD *)(v15 + 8);
            v100 = v110[0];
            v105 = v13;
            v64 = sub_15A9520(v120, v63 >> 8);
            v13 = v105;
            v14 = v100;
            v12 = (unsigned int)(8 * v64);
            break;
        }
        break;
      }
      sub_19CF000(
        (__int64)&v116,
        v14,
        (unsigned __int64)(v12 * v13 + 7) >> 3,
        *(v7 - 6),
        1 << (*((unsigned __int16 *)v7 - 3) >> 1) >> 1,
        (__int64)(v7 - 3));
LABEL_15:
      v7 = (_QWORD *)v7[1];
    }
    else
    {
      if ( (_BYTE)v9 != 78
        || (v16 = *(v7 - 6), *(_BYTE *)(v16 + 16))
        || (*(_BYTE *)(v16 + 33) & 0x20) == 0
        || *(_DWORD *)(v16 + 36) != 137 )
      {
        if ( (unsigned __int8)sub_15F3040((__int64)(v7 - 3)) || (unsigned __int8)sub_15F2ED0((__int64)(v7 - 3)) )
          break;
        goto LABEL_15;
      }
      v17 = *((_DWORD *)v7 - 1) & 0xFFFFFFF;
      v18 = v8[3 * (3 - v17)];
      if ( *(_DWORD *)(v18 + 32) <= 0x40u )
      {
        if ( *(_QWORD *)(v18 + 24) )
          break;
      }
      else
      {
        v96 = *(_DWORD *)(v18 + 32);
        v105 = *((_DWORD *)v7 - 1) & 0xFFFFFFF;
        v19 = sub_16A57B0(v18 + 24);
        v17 = v105;
        if ( v96 != v19 )
          break;
      }
      if ( v107 != v8[3 * (1 - v17)] )
        break;
      if ( *(_BYTE *)(v8[3 * (2 - v17)] + 16LL) != 13 )
        break;
      v20 = sub_1649C60(v8[-3 * v17]);
      if ( !(unsigned __int8)sub_19CFE90(a3, v20, v110, v106) )
        break;
      v21 = v8[3 * (2LL - (*((_DWORD *)v7 - 1) & 0xFFFFFFF))];
      v22 = *(_QWORD **)(v21 + 24);
      if ( *(_DWORD *)(v21 + 32) > 0x40u )
        v22 = (_QWORD *)*v22;
      v90 = (__int64)v22;
      v97 = v110[0];
      LODWORD(v105) = sub_15603A0(v7 + 4, 0);
      v23 = sub_1649C60(v8[-3 * (*((_DWORD *)v7 - 1) & 0xFFFFFFF)]);
      sub_19CF000((__int64)&v116, v97, v90, v23, v105, (__int64)(v7 - 3));
      v7 = (_QWORD *)v7[1];
    }
  }
  if ( !(_DWORD)v117 )
  {
    v105 = 0;
LABEL_67:
    v49 = v116;
    goto LABEL_68;
  }
  if ( v108[16] == 55 )
  {
    v10 = 1;
    v11 = **((_QWORD **)v108 - 6);
    while ( 2 )
    {
      switch ( *(_BYTE *)(v11 + 8) )
      {
        case 1:
          v67 = 16;
          goto LABEL_97;
        case 2:
          v67 = 32;
          goto LABEL_97;
        case 3:
        case 9:
          v67 = 64;
          goto LABEL_97;
        case 4:
          v67 = 80;
          goto LABEL_97;
        case 5:
        case 6:
          v67 = 128;
          goto LABEL_97;
        case 7:
          v67 = 8 * (unsigned int)sub_15A9520(v120, 0);
          goto LABEL_97;
        case 0xB:
          v67 = *(_DWORD *)(v11 + 8) >> 8;
          goto LABEL_97;
        case 0xD:
          v67 = 8LL * *(_QWORD *)sub_15A9930(v120, v11);
          goto LABEL_97;
        case 0xE:
          v92 = v120;
          v87 = *(_QWORD *)(v11 + 24);
          v102 = *(_QWORD *)(v11 + 32);
          v70 = v120;
          v105 = 1;
          v71 = v87;
          v72 = (unsigned int)sub_15A9FE0(v120, v87);
          while ( 2 )
          {
            switch ( *(_BYTE *)(v71 + 8) )
            {
              case 1:
                v76 = 16;
                goto LABEL_126;
              case 2:
                v76 = 32;
                goto LABEL_126;
              case 3:
              case 9:
                v76 = 64;
                goto LABEL_126;
              case 4:
                v76 = 80;
                goto LABEL_126;
              case 5:
              case 6:
                v76 = 128;
                goto LABEL_126;
              case 7:
                v76 = 8 * (unsigned int)sub_15A9520(v92, 0);
                goto LABEL_126;
              case 0xB:
                v76 = *(_DWORD *)(v71 + 8) >> 8;
                goto LABEL_126;
              case 0xD:
                v76 = 8LL * *(_QWORD *)sub_15A9930(v92, v71);
                goto LABEL_126;
              case 0xE:
                v85 = v92;
                v82 = *(_QWORD *)(v71 + 24);
                v95 = *(_QWORD *)(v71 + 32);
                v89 = (unsigned int)sub_15A9FE0(v70, v82);
                v76 = 8 * v95 * v89 * ((v89 + ((unsigned __int64)(sub_127FA20(v85, v82) + 7) >> 3) - 1) / v89);
                goto LABEL_126;
              case 0xF:
                v76 = 8 * (unsigned int)sub_15A9520(v92, *(_DWORD *)(v71 + 8) >> 8);
LABEL_126:
                v67 = 8 * v72 * v102 * ((v72 + ((unsigned __int64)(v105 * v76 + 7) >> 3) - 1) / v72);
                goto LABEL_97;
              case 0x10:
                v77 = v105 * *(_QWORD *)(v71 + 32);
                v71 = *(_QWORD *)(v71 + 24);
                v105 = v77;
                continue;
              default:
                goto LABEL_139;
            }
          }
        case 0xF:
          v67 = 8 * (unsigned int)sub_15A9520(v120, *(_DWORD *)(v11 + 8) >> 8);
LABEL_97:
          sub_19CF000(
            (__int64)&v116,
            0,
            (unsigned __int64)(v67 * v10 + 7) >> 3,
            *((_QWORD *)v108 - 3),
            1 << (*((unsigned __int16 *)v108 + 9) >> 1) >> 1,
            (__int64)v108);
          goto LABEL_36;
        case 0x10:
          v69 = *(_QWORD *)(v11 + 32);
          v11 = *(_QWORD *)(v11 + 24);
          v10 *= v69;
          continue;
        default:
LABEL_139:
          ++v119;
          BUG();
      }
    }
  }
  v24 = *(_QWORD *)&v108[24 * (2LL - (*((_DWORD *)v108 + 5) & 0xFFFFFFF))];
  v25 = *(_QWORD **)(v24 + 24);
  if ( *(_DWORD *)(v24 + 32) > 0x40u )
    v25 = (_QWORD *)*v25;
  v26 = sub_15603A0((_QWORD *)v108 + 7, 0);
  v27 = sub_1649C60(*(_QWORD *)&v108[-24 * (*((_DWORD *)v108 + 5) & 0xFFFFFFF)]);
  sub_19CF000((__int64)&v116, 0, (__int64)v25, v27, v26, (__int64)v108);
LABEL_36:
  v28 = (_QWORD *)sub_16498A0((__int64)(v7 - 3));
  v110[0] = 0;
  v111 = v28;
  v112 = 0;
  v113 = 0;
  v114 = 0;
  v115 = 0;
  v29 = v7[2];
  v110[2] = (__int64)v7;
  v110[1] = v29;
  v30 = (unsigned __int8 *)v7[3];
  v109 = v30;
  if ( v30 )
  {
    sub_1623A60((__int64)&v109, (__int64)v30, 2);
    if ( v110[0] )
      sub_161E7C0((__int64)v110, v110[0]);
    v110[0] = (__int64)v109;
    if ( v109 )
      sub_1623210((__int64)&v109, v109, (__int64)v110);
  }
  v31 = 176LL * (unsigned int)v117;
  v108 = &v116[v31];
  if ( &v116[v31] != v116 )
  {
    v32 = v116;
    v105 = 0;
LABEL_43:
    while ( 1 )
    {
      v33 = *((_DWORD *)v32 + 10);
      if ( v33 != 1 )
      {
        v34 = v33;
        if ( v33 > 3 )
          goto LABEL_50;
        v35 = *((_QWORD *)v32 + 1);
        if ( v35 - *(_QWORD *)v32 > 15 )
          goto LABEL_50;
        if ( v34 > 1 )
          break;
      }
LABEL_60:
      v32 += 176;
      if ( v108 == v32 )
        goto LABEL_61;
    }
    v36 = *((_QWORD *)v32 + 4);
    v37 = v36 + 8 * v34;
    while ( *(_BYTE *)(*(_QWORD *)v36 + 16LL) == 55 )
    {
      v36 += 8;
      if ( v37 == v36 )
      {
        if ( v34 == 2 )
          goto LABEL_60;
        v53 = v35 - *(_QWORD *)v32;
        v54 = sub_15A96E0(v106);
        if ( v54 > 7 )
          v53 = v53 % (v54 >> 3) + v53 / (v54 >> 3);
        if ( *((_DWORD *)v32 + 10) > v53 )
          break;
        v32 += 176;
        if ( v108 != v32 )
          goto LABEL_43;
        goto LABEL_61;
      }
    }
LABEL_50:
    v38 = *((_DWORD *)v32 + 6);
    v39 = (_QWORD *)*((_QWORD *)v32 + 2);
    if ( !v38 )
      v38 = sub_15A9FE0(v106, *(_QWORD *)(*v39 + 24LL));
    v105 = *((_QWORD *)v32 + 1) - *(_QWORD *)v32;
    v40 = sub_1643360(v111);
    v41 = (__int64 *)sub_159C470(v40, v105, 0);
    v105 = (__int64)sub_15E7280(v110, v39, v107, v41, v38, 0, 0, 0, 0);
    v42 = v105;
    if ( !*((_DWORD *)v32 + 10) )
      goto LABEL_60;
    v43 = (unsigned __int8 **)(v105 + 48);
    v44 = *(unsigned __int8 **)(**((_QWORD **)v32 + 4) + 48LL);
    v109 = v44;
    if ( v44 )
    {
      sub_1623A60((__int64)&v109, (__int64)v44, 2);
      if ( v43 == &v109 )
      {
        if ( v109 )
          sub_161E7C0(v105 + 48, (__int64)v109);
        goto LABEL_57;
      }
      v55 = *(_QWORD *)(v105 + 48);
      if ( !v55 )
      {
LABEL_82:
        v56 = v109;
        *(_QWORD *)(v105 + 48) = v109;
        if ( v56 )
          sub_1623210((__int64)&v109, v56, v42 + 48);
        goto LABEL_57;
      }
    }
    else if ( v43 == &v109 || (v55 = *(_QWORD *)(v105 + 48)) == 0 )
    {
LABEL_57:
      v45 = (_QWORD **)*((_QWORD *)v32 + 4);
      v46 = &v45[*((unsigned int *)v32 + 10)];
      if ( v45 != v46 )
      {
        v47 = (_QWORD **)*((_QWORD *)v32 + 4);
        do
        {
          v48 = *v47++;
          sub_14191F0(*a1, (__int64)v48);
          sub_15F20C0(v48);
        }
        while ( v46 != v47 );
      }
      goto LABEL_60;
    }
    sub_161E7C0(v105 + 48, v55);
    goto LABEL_82;
  }
  v105 = 0;
LABEL_61:
  if ( v110[0] )
    sub_161E7C0((__int64)v110, v110[0]);
  v49 = v116;
  v50 = &v116[176 * (unsigned int)v117];
  if ( v116 != v50 )
  {
    do
    {
      v50 -= 176;
      v51 = *((_QWORD *)v50 + 4);
      if ( (_BYTE *)v51 != v50 + 48 )
        _libc_free(v51);
    }
    while ( v49 != v50 );
    goto LABEL_67;
  }
LABEL_68:
  if ( v49 != v118 )
    _libc_free((unsigned __int64)v49);
  return v105;
}
