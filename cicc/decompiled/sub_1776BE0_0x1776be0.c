// Function: sub_1776BE0
// Address: 0x1776be0
//
__int64 __fastcall sub_1776BE0(__int64 *a1, __int64 a2, __int64 a3, unsigned int *a4, int a5, int a6)
{
  unsigned int v6; // r13d
  unsigned int v8; // r12d
  __int64 v9; // r14
  __int64 v10; // rdi
  unsigned int v11; // r15d
  unsigned int v12; // eax
  unsigned int v13; // r13d
  __int64 v15; // rdx
  __int64 v16; // rax
  _QWORD *v17; // r12
  _QWORD *v18; // r13
  __int64 v19; // rax
  unsigned __int64 v20; // r14
  _QWORD *v21; // rax
  int v22; // edx
  _BYTE *v23; // rsi
  __int64 v24; // rdi
  __int64 v25; // rax
  __int64 v26; // r13
  unsigned __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // r14
  unsigned int v30; // r12d
  __int64 v31; // rax
  __int64 i; // rax
  __int64 v33; // rdi
  int v34; // r8d
  int v35; // r9d
  __int64 v36; // r12
  __int64 *v37; // rax
  char v38; // dl
  unsigned __int8 v39; // al
  __int64 v40; // r12
  _QWORD *v41; // rdi
  __int64 v42; // rax
  unsigned int v43; // r14d
  unsigned int v44; // eax
  __int64 v45; // rcx
  unsigned int v46; // edx
  __int64 v47; // rcx
  __int64 v48; // rax
  unsigned __int64 v49; // r13
  __int64 *v50; // rsi
  __int64 *v51; // rcx
  __int64 v52; // rdi
  unsigned __int64 v53; // rax
  __int64 v54; // r13
  __int64 v55; // rsi
  __int64 v56; // r12
  unsigned __int64 v57; // r14
  __int64 v58; // rax
  unsigned int v59; // r12d
  bool v60; // r12
  __int64 v61; // rax
  __int64 v62; // r13
  __int64 v63; // rax
  __int64 v64; // rax
  __int64 *v65; // r13
  __int64 v66; // r14
  char v67; // al
  __int64 v68; // r12
  __int64 v69; // r13
  unsigned __int64 v70; // r14
  __int64 v71; // rax
  __int64 v72; // rsi
  __int64 v73; // rax
  __int64 v74; // r12
  __int64 v75; // rax
  __int64 v76; // rax
  __int64 v77; // rdx
  unsigned __int64 *v78; // rdi
  __int64 v79; // [rsp+8h] [rbp-168h]
  unsigned __int64 v80; // [rsp+10h] [rbp-160h]
  __int64 v81; // [rsp+18h] [rbp-158h]
  __int64 v82; // [rsp+18h] [rbp-158h]
  unsigned __int64 v83; // [rsp+20h] [rbp-150h]
  unsigned __int64 v84; // [rsp+20h] [rbp-150h]
  __int64 v85; // [rsp+28h] [rbp-148h]
  __int64 v86; // [rsp+28h] [rbp-148h]
  unsigned __int64 v87; // [rsp+30h] [rbp-140h]
  __int64 v88; // [rsp+30h] [rbp-140h]
  __int64 v89; // [rsp+38h] [rbp-138h]
  __int64 v90; // [rsp+40h] [rbp-130h]
  unsigned int v94; // [rsp+58h] [rbp-118h]
  __int64 v95; // [rsp+60h] [rbp-110h] BYREF
  unsigned int v96; // [rsp+68h] [rbp-108h]
  __int64 v97; // [rsp+70h] [rbp-100h] BYREF
  unsigned int v98; // [rsp+78h] [rbp-F8h]
  unsigned __int64 *v99; // [rsp+80h] [rbp-F0h] BYREF
  unsigned int v100; // [rsp+88h] [rbp-E8h]
  _BYTE *v101; // [rsp+90h] [rbp-E0h] BYREF
  __int64 v102; // [rsp+98h] [rbp-D8h]
  _BYTE v103[32]; // [rsp+A0h] [rbp-D0h] BYREF
  _QWORD *v104; // [rsp+C0h] [rbp-B0h] BYREF
  unsigned int v105; // [rsp+C8h] [rbp-A8h]
  unsigned int v106; // [rsp+CCh] [rbp-A4h]
  _QWORD v107[4]; // [rsp+D0h] [rbp-A0h] BYREF
  __int64 v108; // [rsp+F0h] [rbp-80h] BYREF
  __int64 *v109; // [rsp+F8h] [rbp-78h]
  __int64 *v110; // [rsp+100h] [rbp-70h]
  __int64 v111; // [rsp+108h] [rbp-68h]
  int v112; // [rsp+110h] [rbp-60h]
  _BYTE v113[88]; // [rsp+118h] [rbp-58h] BYREF

  v6 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  if ( v6 <= 1 )
    return 0;
  v8 = 1;
  v9 = a2 - 24LL * v6;
  do
  {
    v10 = *(_QWORD *)(v9 + 24);
    if ( *(_BYTE *)(v10 + 16) != 13 )
      break;
    v11 = *(_DWORD *)(v10 + 32);
    if ( v11 <= 0x40 )
    {
      if ( *(_QWORD *)(v10 + 24) )
        break;
    }
    else if ( v11 != (unsigned int)sub_16A57B0(v10 + 24) )
    {
      break;
    }
    ++v8;
    v9 += 24;
  }
  while ( v6 != v8 );
  *a4 = v8;
  v12 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  if ( v12 == v8 || *(_BYTE *)(*(_QWORD *)(a2 + 24 * (v8 - (unsigned __int64)v12)) + 16LL) <= 0x10u )
    return 0;
  v101 = v103;
  v102 = 0x400000000LL;
  v15 = 24 * (1LL - v12);
  v16 = 24 * (1LL - v12 + v8);
  v17 = (_QWORD *)(a2 + v15);
  v18 = (_QWORD *)(a2 + v16);
  v19 = v16 - v15;
  v20 = 0xAAAAAAAAAAAAAAABLL * (v19 >> 3);
  if ( (unsigned __int64)v19 > 0x60 )
  {
    sub_16CD150((__int64)&v101, v103, 0xAAAAAAAAAAAAAAABLL * (v19 >> 3), 8, a5, a6);
    v23 = v101;
    v22 = v102;
    v21 = &v101[8 * (unsigned int)v102];
  }
  else
  {
    v21 = v103;
    v22 = 0;
    v23 = v103;
  }
  if ( v18 != v17 )
  {
    do
    {
      if ( v21 )
        *v21 = *v17;
      v17 += 3;
      ++v21;
    }
    while ( v18 != v17 );
    v23 = v101;
    v22 = v102;
  }
  v24 = *(_QWORD *)(a2 + 56);
  LODWORD(v102) = v20 + v22;
  v25 = sub_15F9F50(v24, (__int64)v23, (unsigned int)(v20 + v22));
  v26 = v25;
  if ( !v25 )
    goto LABEL_22;
  v27 = *(unsigned __int8 *)(v25 + 8);
  if ( (unsigned __int8)v27 > 0xFu || (v28 = 35454, !_bittest64(&v28, v27)) )
  {
    if ( (unsigned int)(v27 - 13) > 1 && (_DWORD)v27 != 16 || !sub_16435F0(v26, 0) )
      goto LABEL_22;
  }
  v29 = 1;
  v90 = a1[333];
  v30 = sub_15A9FE0(v90, v26);
  while ( 2 )
  {
    switch ( *(_BYTE *)(v26 + 8) )
    {
      case 0:
      case 8:
      case 0xA:
      case 0xC:
      case 0x10:
        v48 = *(_QWORD *)(v26 + 32);
        v26 = *(_QWORD *)(v26 + 24);
        v29 *= v48;
        continue;
      case 1:
        v31 = 16;
        break;
      case 2:
        v31 = 32;
        break;
      case 3:
      case 9:
        v31 = 64;
        break;
      case 4:
        v31 = 80;
        break;
      case 5:
      case 6:
        v31 = 128;
        break;
      case 7:
        v31 = 8 * (unsigned int)sub_15A9520(v90, 0);
        break;
      case 0xB:
        v31 = *(_DWORD *)(v26 + 8) >> 8;
        break;
      case 0xD:
        v31 = 8LL * *(_QWORD *)sub_15A9930(v90, v26);
        break;
      case 0xE:
        v88 = *(_QWORD *)(v26 + 24);
        v89 = *(_QWORD *)(v26 + 32);
        v49 = (unsigned int)sub_15A9FE0(v90, v88);
        v31 = 8 * v49 * v89 * ((v49 + ((unsigned __int64)(sub_127FA20(v90, v88) + 7) >> 3) - 1) / v49);
        break;
      case 0xF:
        v31 = 8 * (unsigned int)sub_15A9520(v90, *(_DWORD *)(v26 + 8) >> 8);
        break;
    }
    break;
  }
  v87 = v30 * ((v30 + ((unsigned __int64)(v29 * v31 + 7) >> 3) - 1) / v30);
  i = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  if ( *a4 + 1 != (_DWORD)i )
  {
    if ( !sub_15FA300(a2) )
      goto LABEL_22;
    i = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  }
  v108 = 0;
  v111 = 4;
  v33 = *(_QWORD *)(a2 - 24 * i);
  v112 = 0;
  v106 = 4;
  v107[0] = v33;
  v109 = (__int64 *)v113;
  v110 = (__int64 *)v113;
  v104 = v107;
  LODWORD(i) = 1;
  while ( 1 )
  {
    v105 = i - 1;
    v36 = sub_1649C60(v33);
    v37 = v109;
    if ( v110 != v109 )
      goto LABEL_38;
    v50 = &v109[HIDWORD(v111)];
    if ( v109 != v50 )
    {
      v51 = 0;
      while ( v36 != *v37 )
      {
        if ( *v37 == -2 )
          v51 = v37;
        if ( v50 == ++v37 )
        {
          if ( !v51 )
            goto LABEL_157;
          *v51 = v36;
          --v112;
          ++v108;
          goto LABEL_39;
        }
      }
LABEL_82:
      LODWORD(i) = v105;
      goto LABEL_45;
    }
LABEL_157:
    if ( HIDWORD(v111) < (unsigned int)v111 )
    {
      ++HIDWORD(v111);
      *v50 = v36;
      ++v108;
    }
    else
    {
LABEL_38:
      sub_16CCBA0((__int64)&v108, v36);
      if ( !v38 )
        goto LABEL_82;
    }
LABEL_39:
    v39 = *(_BYTE *)(v36 + 16);
    if ( v39 <= 0x17u )
    {
      if ( v39 == 1 )
      {
        switch ( *(_BYTE *)(v36 + 32) & 0xF )
        {
          case 0:
          case 1:
          case 3:
          case 5:
          case 6:
          case 7:
          case 8:
            v40 = *(_QWORD *)(v36 - 24);
            i = v105;
            if ( v105 < v106 )
              goto LABEL_44;
            goto LABEL_119;
          case 2:
          case 4:
          case 9:
          case 0xA:
            goto LABEL_49;
          default:
            goto LABEL_42;
        }
      }
      if ( v39 == 3 && !sub_15E4F60(v36) )
      {
        switch ( *(_BYTE *)(v36 + 32) & 0xF )
        {
          case 0:
          case 1:
          case 3:
          case 5:
          case 6:
          case 7:
          case 8:
            v67 = *(_BYTE *)(v36 + 80);
            if ( (v67 & 2) == 0 && (v67 & 1) != 0 )
            {
              v68 = *(_QWORD *)(v36 + 24);
              v69 = 1;
              v70 = (unsigned int)sub_15A9FE0(v90, v68);
              while ( 2 )
              {
                switch ( *(_BYTE *)(v68 + 8) )
                {
                  case 1:
                    v42 = 16;
                    goto LABEL_48;
                  case 2:
                    v42 = 32;
                    goto LABEL_48;
                  case 3:
                  case 9:
                    v42 = 64;
                    goto LABEL_48;
                  case 4:
                    v42 = 80;
                    goto LABEL_48;
                  case 5:
                  case 6:
                    v42 = 128;
                    goto LABEL_48;
                  case 7:
                    v42 = 8 * (unsigned int)sub_15A9520(v90, 0);
                    goto LABEL_48;
                  case 0xB:
                    v42 = *(_DWORD *)(v68 + 8) >> 8;
                    goto LABEL_48;
                  case 0xD:
                    v42 = 8LL * *(_QWORD *)sub_15A9930(v90, v68);
                    goto LABEL_48;
                  case 0xE:
                    v72 = *(_QWORD *)(v68 + 24);
                    v73 = *(_QWORD *)(v68 + 32);
                    v74 = 1;
                    v86 = v73;
                    v84 = (unsigned int)sub_15A9FE0(v90, v72);
                    while ( 2 )
                    {
                      switch ( *(_BYTE *)(v72 + 8) )
                      {
                        case 1:
                          v76 = 16;
                          goto LABEL_147;
                        case 2:
                          v76 = 32;
                          goto LABEL_147;
                        case 3:
                        case 9:
                          v76 = 64;
                          goto LABEL_147;
                        case 4:
                          v76 = 80;
                          goto LABEL_147;
                        case 5:
                        case 6:
                          v76 = 128;
                          goto LABEL_147;
                        case 7:
                          v76 = 8 * (unsigned int)sub_15A9520(v90, 0);
                          goto LABEL_147;
                        case 0xB:
                          v76 = *(_DWORD *)(v72 + 8) >> 8;
                          goto LABEL_147;
                        case 0xD:
                          v76 = 8LL * *(_QWORD *)sub_15A9930(v90, v72);
                          goto LABEL_147;
                        case 0xE:
                          v82 = *(_QWORD *)(v72 + 32);
                          v79 = *(_QWORD *)(v72 + 24);
                          v80 = (unsigned int)sub_15A9FE0(v90, v79);
                          v76 = 8 * v82 * v80 * ((v80 + ((unsigned __int64)(sub_127FA20(v90, v79) + 7) >> 3) - 1) / v80);
                          goto LABEL_147;
                        case 0xF:
                          v76 = 8 * (unsigned int)sub_15A9520(v90, *(_DWORD *)(v72 + 8) >> 8);
LABEL_147:
                          v42 = 8 * v84 * v86 * ((v84 + ((unsigned __int64)(v74 * v76 + 7) >> 3) - 1) / v84);
                          goto LABEL_48;
                        case 0x10:
                          v75 = *(_QWORD *)(v72 + 32);
                          v72 = *(_QWORD *)(v72 + 24);
                          v74 *= v75;
                          continue;
                        default:
                          goto LABEL_42;
                      }
                    }
                  case 0xF:
                    v42 = 8 * (unsigned int)sub_15A9520(v90, *(_DWORD *)(v68 + 8) >> 8);
LABEL_48:
                    if ( v87 < v70 * ((v70 + ((unsigned __int64)(v42 * v69 + 7) >> 3) - 1) / v70) )
                      goto LABEL_49;
                    goto LABEL_82;
                  case 0x10:
                    v71 = *(_QWORD *)(v68 + 32);
                    v68 = *(_QWORD *)(v68 + 24);
                    v69 *= v71;
                    continue;
                  default:
                    goto LABEL_42;
                }
              }
            }
            break;
          case 2:
          case 4:
          case 9:
          case 0xA:
            break;
          default:
            goto LABEL_42;
        }
      }
LABEL_49:
      v41 = v104;
      v13 = 0;
      goto LABEL_50;
    }
    if ( v39 == 79 )
    {
      v62 = *(_QWORD *)(v36 - 48);
      v63 = v105;
      if ( v105 >= v106 )
      {
        sub_16CD150((__int64)&v104, v107, 0, 8, v34, v35);
        v63 = v105;
      }
      v104[v63] = v62;
      i = v105 + 1;
      v105 = i;
      v40 = *(_QWORD *)(v36 - 24);
      if ( v106 <= (unsigned int)i )
      {
LABEL_119:
        sub_16CD150((__int64)&v104, v107, 0, 8, v34, v35);
        i = v105;
      }
LABEL_44:
      v104[i] = v40;
      LODWORD(i) = ++v105;
    }
    else
    {
      if ( v39 != 77 )
      {
        if ( v39 == 53 )
        {
          if ( (v52 = *(_QWORD *)(v36 + 56), v53 = *(unsigned __int8 *)(v52 + 8), (unsigned __int8)v53 <= 0xFu)
            && (v77 = 35454, _bittest64(&v77, v53))
            || ((unsigned int)(v53 - 13) <= 1 || (_DWORD)v53 == 16) && sub_16435F0(v52, 0) )
          {
            v54 = *(_QWORD *)(v36 - 24);
            if ( *(_BYTE *)(v54 + 16) == 13 )
            {
              v55 = *(_QWORD *)(v36 + 56);
              v56 = 1;
              v57 = (unsigned int)sub_15A9FE0(v90, v55);
              while ( 2 )
              {
                switch ( *(_BYTE *)(v55 + 8) )
                {
                  case 1:
                    v58 = 16;
                    goto LABEL_94;
                  case 2:
                    v58 = 32;
                    goto LABEL_94;
                  case 3:
                  case 9:
                    v58 = 64;
                    goto LABEL_94;
                  case 4:
                    v58 = 80;
                    goto LABEL_94;
                  case 5:
                  case 6:
                    v58 = 128;
                    goto LABEL_94;
                  case 7:
                    v58 = 8 * (unsigned int)sub_15A9520(v90, 0);
                    goto LABEL_94;
                  case 0xB:
                    v58 = *(_DWORD *)(v55 + 8) >> 8;
                    goto LABEL_94;
                  case 0xD:
                    v58 = 8LL * *(_QWORD *)sub_15A9930(v90, v55);
                    goto LABEL_94;
                  case 0xE:
                    v85 = *(_QWORD *)(v55 + 32);
                    v81 = *(_QWORD *)(v55 + 24);
                    v83 = (unsigned int)sub_15A9FE0(v90, v81);
                    v58 = 8 * v85 * v83 * ((v83 + ((unsigned __int64)(sub_127FA20(v90, v81) + 7) >> 3) - 1) / v83);
                    goto LABEL_94;
                  case 0xF:
                    v58 = 8 * (unsigned int)sub_15A9520(v90, *(_DWORD *)(v55 + 8) >> 8);
LABEL_94:
                    v98 = 128;
                    sub_16A4EF0((__int64)&v97, v57 * ((v57 + ((unsigned __int64)(v56 * v58 + 7) >> 3) - 1) / v57), 0);
                    sub_16A5DD0((__int64)&v95, v54 + 24, 0x80u);
                    sub_16A7B50((__int64)&v99, (__int64)&v95, &v97);
                    v59 = v100;
                    if ( v100 <= 0x40 )
                    {
                      v60 = v87 < (unsigned __int64)v99;
                      goto LABEL_96;
                    }
                    if ( v59 - (unsigned int)sub_16A57B0((__int64)&v99) <= 0x40 )
                    {
                      v78 = v99;
                      v60 = v87 < *v99;
                    }
                    else
                    {
                      v78 = v99;
                      v60 = 1;
                      if ( !v99 )
                        goto LABEL_96;
                    }
                    j_j___libc_free_0_0(v78);
LABEL_96:
                    if ( v96 > 0x40 && v95 )
                      j_j___libc_free_0_0(v95);
                    if ( v98 > 0x40 && v97 )
                      j_j___libc_free_0_0(v97);
                    if ( !v60 )
                      goto LABEL_82;
                    goto LABEL_49;
                  case 0x10:
                    v61 = *(_QWORD *)(v55 + 32);
                    v55 = *(_QWORD *)(v55 + 24);
                    v56 *= v61;
                    continue;
                  default:
LABEL_42:
                    BUG();
                }
              }
            }
          }
        }
        goto LABEL_49;
      }
      v64 = 3LL * (*(_DWORD *)(v36 + 20) & 0xFFFFFFF);
      if ( (*(_BYTE *)(v36 + 23) & 0x40) != 0 )
      {
        v65 = *(__int64 **)(v36 - 8);
        v36 = (__int64)&v65[v64];
      }
      else
      {
        v65 = (__int64 *)(v36 - v64 * 8);
      }
      for ( i = v105; (__int64 *)v36 != v65; ++v105 )
      {
        v66 = *v65;
        if ( v106 <= (unsigned int)i )
        {
          sub_16CD150((__int64)&v104, v107, 0, 8, v34, v35);
          i = v105;
        }
        v65 += 3;
        v104[i] = v66;
        i = v105 + 1;
      }
    }
LABEL_45:
    v41 = v104;
    if ( !(_DWORD)i )
      break;
    v33 = v104[(unsigned int)i - 1];
  }
  v13 = 1;
LABEL_50:
  if ( v41 != v107 )
    _libc_free((unsigned __int64)v41);
  if ( v110 != v109 )
    _libc_free((unsigned __int64)v110);
  if ( (_BYTE)v13 )
  {
    v43 = *a4 + 1;
    v44 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
    if ( v43 != v44 )
    {
      v45 = v44;
      v94 = v44 - 1;
      while ( 1 )
      {
        sub_14C2530((__int64)&v108, *(__int64 **)(a2 + 24 * (v43 - v45)), a1[333], 0, a1[330], a3, a1[332], 0);
        v46 = (unsigned int)v109;
        v47 = (unsigned int)v109 > 0x40 ? *(_QWORD *)(v108 + 8LL * ((unsigned int)((_DWORD)v109 - 1) >> 6)) : v108;
        if ( (v47 & (1LL << ((unsigned __int8)v109 - 1))) == 0 )
          break;
        if ( (unsigned int)v111 > 0x40 && v110 )
        {
          j_j___libc_free_0_0(v110);
          v46 = (unsigned int)v109;
        }
        if ( v46 > 0x40 && v108 )
          j_j___libc_free_0_0(v108);
        if ( v43 == v94 )
        {
          v13 = (unsigned __int8)v13;
          goto LABEL_23;
        }
        ++v43;
        v45 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
      }
      if ( (unsigned int)v111 > 0x40 && v110 )
      {
        j_j___libc_free_0_0(v110);
        v46 = (unsigned int)v109;
      }
      if ( v46 > 0x40 && v108 )
        j_j___libc_free_0_0(v108);
      goto LABEL_22;
    }
  }
  else
  {
LABEL_22:
    v13 = 0;
  }
LABEL_23:
  if ( v101 != v103 )
    _libc_free((unsigned __int64)v101);
  return v13;
}
