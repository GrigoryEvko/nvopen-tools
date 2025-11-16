// Function: sub_38530B0
// Address: 0x38530b0
//
__int64 __fastcall sub_38530B0(__int64 a1, __int64 a2)
{
  int v4; // eax
  __int64 v5; // rcx
  int v6; // edi
  __int64 v7; // rsi
  unsigned int v8; // edx
  __int64 *v9; // rax
  __int64 v10; // r8
  __int64 v11; // r14
  __int64 v12; // rbx
  __int64 v13; // rsi
  __int64 v14; // rdi
  __int64 v15; // r8
  unsigned __int64 v16; // r9
  unsigned __int64 v17; // r15
  __int64 result; // rax
  __int64 v19; // r8
  int v20; // r9d
  __int64 v21; // r14
  __int64 v22; // r15
  __int64 v23; // rbx
  unsigned __int64 v24; // r14
  __int64 v25; // rdi
  unsigned __int64 v26; // r8
  __int64 v27; // rax
  unsigned __int64 v28; // rax
  unsigned int v29; // r15d
  unsigned __int64 v30; // rbx
  unsigned __int64 v31; // rcx
  unsigned __int64 v32; // rsi
  int v33; // esi
  unsigned __int64 v34; // rdx
  unsigned int v35; // eax
  unsigned __int64 v36; // rdx
  __int64 v37; // rax
  unsigned __int64 v38; // rax
  unsigned __int64 v39; // rcx
  __int64 v40; // rax
  unsigned __int64 v41; // rax
  unsigned __int64 v42; // rdx
  unsigned int v43; // eax
  __int64 v44; // rsi
  __int64 v45; // rdi
  __int64 v46; // r9
  unsigned __int64 v47; // r10
  __int64 v48; // rax
  _QWORD *v49; // rax
  unsigned int v50; // eax
  __int64 v51; // rsi
  __int64 v52; // r10
  __int64 v53; // rdi
  unsigned __int64 v54; // r15
  int v55; // eax
  __int64 v56; // rax
  int v57; // eax
  __int64 v58; // rax
  __int64 v59; // rax
  __int64 v60; // rax
  _QWORD *v61; // rax
  int v62; // eax
  int v63; // eax
  __int64 v64; // rax
  __int64 v65; // rax
  int v66; // eax
  __int64 v67; // rax
  int v68; // eax
  _QWORD *v69; // rax
  int v70; // eax
  int v71; // r9d
  __int64 v72; // [rsp+0h] [rbp-70h]
  unsigned __int64 v73; // [rsp+8h] [rbp-68h]
  __int64 v74; // [rsp+8h] [rbp-68h]
  __int64 v75; // [rsp+10h] [rbp-60h]
  __int64 v76; // [rsp+10h] [rbp-60h]
  __int64 v77; // [rsp+18h] [rbp-58h]
  __int64 v78; // [rsp+18h] [rbp-58h]
  __int64 v79; // [rsp+20h] [rbp-50h]
  __int64 v80; // [rsp+20h] [rbp-50h]
  unsigned __int64 v81; // [rsp+20h] [rbp-50h]
  __int64 v82; // [rsp+20h] [rbp-50h]
  __int64 v83; // [rsp+20h] [rbp-50h]
  __int64 v84; // [rsp+20h] [rbp-50h]
  __int64 v85; // [rsp+20h] [rbp-50h]
  unsigned __int64 v86; // [rsp+28h] [rbp-48h]
  __int64 v87; // [rsp+28h] [rbp-48h]
  __int64 v88; // [rsp+28h] [rbp-48h]
  unsigned __int64 v89; // [rsp+28h] [rbp-48h]
  __int64 v90; // [rsp+28h] [rbp-48h]
  __int64 v91; // [rsp+28h] [rbp-48h]
  __int64 v92; // [rsp+30h] [rbp-40h]
  __int64 v93; // [rsp+30h] [rbp-40h]
  __int64 v94; // [rsp+30h] [rbp-40h]
  unsigned __int64 v95; // [rsp+30h] [rbp-40h]
  unsigned __int64 v96; // [rsp+30h] [rbp-40h]
  unsigned __int64 v97; // [rsp+30h] [rbp-40h]
  __int64 v98; // [rsp+30h] [rbp-40h]
  __int64 v99; // [rsp+38h] [rbp-38h]
  __int64 v100; // [rsp+38h] [rbp-38h]
  __int64 v101; // [rsp+38h] [rbp-38h]
  unsigned __int64 v102; // [rsp+38h] [rbp-38h]
  __int64 v103; // [rsp+38h] [rbp-38h]
  unsigned __int64 v104; // [rsp+38h] [rbp-38h]
  unsigned __int64 v105; // [rsp+38h] [rbp-38h]

  if ( !(unsigned __int8)sub_15F8BF0(a2) )
    goto LABEL_8;
  v4 = *(_DWORD *)(a1 + 160);
  if ( !v4 )
    goto LABEL_8;
  v5 = *(_QWORD *)(a2 - 24);
  v6 = v4 - 1;
  v7 = *(_QWORD *)(a1 + 144);
  v8 = (v4 - 1) & (((unsigned int)*(_QWORD *)(a2 - 24) >> 9) ^ ((unsigned int)v5 >> 4));
  v9 = (__int64 *)(v7 + 16LL * v8);
  v10 = *v9;
  if ( v5 != *v9 )
  {
    v70 = 1;
    while ( v10 != -8 )
    {
      v71 = v70 + 1;
      v8 = v6 & (v70 + v8);
      v9 = (__int64 *)(v7 + 16LL * v8);
      v10 = *v9;
      if ( v5 == *v9 )
        goto LABEL_4;
      v70 = v71;
    }
LABEL_8:
    if ( (unsigned __int8)sub_15F8F00(a2) )
    {
      v22 = *(_QWORD *)(a2 + 56);
      v23 = 1;
      v24 = *(_QWORD *)(a1 + 104);
      v100 = *(_QWORD *)(a1 + 40);
      v25 = v100;
      v26 = (unsigned int)sub_15A9FE0(v100, v22);
      while ( 2 )
      {
        switch ( *(_BYTE *)(v22 + 8) )
        {
          case 1:
            v40 = 16;
            goto LABEL_42;
          case 2:
            v40 = 32;
            goto LABEL_42;
          case 3:
          case 9:
            v40 = 64;
            goto LABEL_42;
          case 4:
            v40 = 80;
            goto LABEL_42;
          case 5:
          case 6:
            v40 = 128;
            goto LABEL_42;
          case 7:
            v105 = v26;
            v57 = sub_15A9520(v25, 0);
            v26 = v105;
            v40 = (unsigned int)(8 * v57);
            goto LABEL_42;
          case 0xB:
            v40 = *(_DWORD *)(v22 + 8) >> 8;
            goto LABEL_42;
          case 0xD:
            v102 = v26;
            v49 = (_QWORD *)sub_15A9930(v25, v22);
            v26 = v102;
            v40 = 8LL * *v49;
            goto LABEL_42;
          case 0xE:
            v80 = v26;
            v93 = v100;
            v87 = *(_QWORD *)(v22 + 24);
            v103 = *(_QWORD *)(v22 + 32);
            v50 = sub_15A9FE0(v25, v87);
            v26 = v80;
            v51 = v87;
            v52 = 1;
            v53 = v93;
            v54 = v50;
            while ( 2 )
            {
              switch ( *(_BYTE *)(v51 + 8) )
              {
                case 1:
                  v59 = 16;
                  goto LABEL_74;
                case 2:
                  v59 = 32;
                  goto LABEL_74;
                case 3:
                case 9:
                  v59 = 64;
                  goto LABEL_74;
                case 4:
                  v59 = 80;
                  goto LABEL_74;
                case 5:
                case 6:
                  v59 = 128;
                  goto LABEL_74;
                case 7:
                  v90 = v52;
                  v68 = sub_15A9520(v93, 0);
                  v26 = v80;
                  v52 = v90;
                  v59 = (unsigned int)(8 * v68);
                  goto LABEL_74;
                case 0xB:
                  v59 = *(_DWORD *)(v51 + 8) >> 8;
                  goto LABEL_74;
                case 0xD:
                  v91 = v52;
                  v69 = (_QWORD *)sub_15A9930(v93, v51);
                  v26 = v80;
                  v52 = v91;
                  v59 = 8LL * *v69;
                  goto LABEL_74;
                case 0xE:
                  v74 = v52;
                  v76 = v80;
                  v78 = *(_QWORD *)(v51 + 24);
                  v85 = v93;
                  v98 = *(_QWORD *)(v51 + 32);
                  v89 = (unsigned int)sub_15A9FE0(v53, v78);
                  v67 = sub_127FA20(v85, v78);
                  v26 = v76;
                  v52 = v74;
                  v59 = 8 * v98 * v89 * ((v89 + ((unsigned __int64)(v67 + 7) >> 3) - 1) / v89);
                  goto LABEL_74;
                case 0xF:
                  v88 = v52;
                  v66 = sub_15A9520(v93, *(_DWORD *)(v51 + 8) >> 8);
                  v26 = v80;
                  v52 = v88;
                  v59 = (unsigned int)(8 * v66);
LABEL_74:
                  v40 = 8 * v54 * v103 * ((v54 + ((unsigned __int64)(v59 * v52 + 7) >> 3) - 1) / v54);
                  goto LABEL_42;
                case 0x10:
                  v65 = *(_QWORD *)(v51 + 32);
                  v51 = *(_QWORD *)(v51 + 24);
                  v52 *= v65;
                  continue;
                default:
                  goto LABEL_98;
              }
            }
          case 0xF:
            v104 = v26;
            v55 = sub_15A9520(v25, *(_DWORD *)(v22 + 8) >> 8);
            v26 = v104;
            v40 = (unsigned int)(8 * v55);
LABEL_42:
            v41 = v26 * ((v26 + ((unsigned __int64)(v40 * v23 + 7) >> 3) - 1) / v26);
            v42 = v24 + v41;
            if ( v24 >= v41 )
              v41 = v24;
            if ( v42 < v41 )
              v42 = -1;
            *(_QWORD *)(a1 + 104) = v42;
            result = sub_15F8F00(a2);
            if ( (_BYTE)result )
              goto LABEL_11;
            goto LABEL_10;
          case 0x10:
            v56 = *(_QWORD *)(v22 + 32);
            v22 = *(_QWORD *)(v22 + 24);
            v23 *= v56;
            continue;
          default:
LABEL_98:
            ++*(_DWORD *)(v24 + 248);
            BUG();
        }
      }
    }
    result = sub_15F8F00(a2);
    if ( !(_BYTE)result )
    {
LABEL_10:
      *(_BYTE *)(a1 + 84) = 1;
      return result;
    }
LABEL_11:
    v21 = *(_QWORD *)(a2 - 24);
    if ( !(unsigned __int8)sub_3852EE0(a1, a2, a2, a1, v19, v20) )
      goto LABEL_30;
    return 1;
  }
LABEL_4:
  v11 = v9[1];
  if ( !v11 || *(_BYTE *)(v11 + 16) != 13 )
    goto LABEL_8;
  v12 = 1;
  v99 = *(_QWORD *)(a1 + 40);
  v86 = *(_QWORD *)(a1 + 104);
  v13 = *(_QWORD *)(a2 + 56);
  v14 = v99;
  v17 = (unsigned int)sub_15A9FE0(v99, v13);
  while ( 2 )
  {
    switch ( *(_BYTE *)(v13 + 8) )
    {
      case 0:
      case 8:
      case 0xA:
      case 0xC:
      case 0x10:
        v48 = *(_QWORD *)(v13 + 32);
        v13 = *(_QWORD *)(v13 + 24);
        v12 *= v48;
        continue;
      case 1:
        v27 = 16;
        break;
      case 2:
        v27 = 32;
        break;
      case 3:
      case 9:
        v27 = 64;
        break;
      case 4:
        v27 = 80;
        break;
      case 5:
      case 6:
        v27 = 128;
        break;
      case 7:
        v27 = 8 * (unsigned int)sub_15A9520(v99, 0);
        break;
      case 0xB:
        v27 = *(_DWORD *)(v13 + 8) >> 8;
        break;
      case 0xD:
        v27 = 8LL * *(_QWORD *)sub_15A9930(v99, v13);
        break;
      case 0xE:
        v92 = v99;
        v79 = *(_QWORD *)(v13 + 24);
        v101 = *(_QWORD *)(v13 + 32);
        v43 = sub_15A9FE0(v14, v79);
        v44 = v79;
        v45 = v92;
        v46 = 1;
        v47 = v43;
        while ( 2 )
        {
          switch ( *(_BYTE *)(v44 + 8) )
          {
            case 0:
            case 8:
            case 0xA:
            case 0xC:
            case 0x10:
              v64 = *(_QWORD *)(v44 + 32);
              v44 = *(_QWORD *)(v44 + 24);
              v46 *= v64;
              continue;
            case 1:
              v58 = 16;
              break;
            case 2:
              v58 = 32;
              break;
            case 3:
            case 9:
              v58 = 64;
              break;
            case 4:
              v58 = 80;
              break;
            case 5:
            case 6:
              v58 = 128;
              break;
            case 7:
              v83 = v46;
              v96 = v47;
              v62 = sub_15A9520(v45, 0);
              v47 = v96;
              v46 = v83;
              v58 = (unsigned int)(8 * v62);
              break;
            case 0xB:
              v58 = *(_DWORD *)(v44 + 8) >> 8;
              break;
            case 0xD:
              v82 = v46;
              v95 = v47;
              v61 = (_QWORD *)sub_15A9930(v45, v44);
              v47 = v95;
              v46 = v82;
              v58 = 8LL * *v61;
              break;
            case 0xE:
              v72 = v46;
              v73 = v47;
              v75 = *(_QWORD *)(v44 + 24);
              v77 = v92;
              v94 = *(_QWORD *)(v44 + 32);
              v81 = (unsigned int)sub_15A9FE0(v45, v75);
              v60 = sub_127FA20(v77, v75);
              v47 = v73;
              v46 = v72;
              v58 = 8 * v94 * v81 * ((v81 + ((unsigned __int64)(v60 + 7) >> 3) - 1) / v81);
              break;
            case 0xF:
              v84 = v46;
              v97 = v47;
              v63 = sub_15A9520(v45, *(_DWORD *)(v44 + 8) >> 8);
              v47 = v97;
              v46 = v84;
              v58 = (unsigned int)(8 * v63);
              break;
          }
          break;
        }
        v16 = (unsigned __int64)(v58 * v46 + 7) >> 3;
        v27 = 8 * v101 * v47 * ((v47 + v16 - 1) / v47);
        break;
      case 0xF:
        v27 = 8 * (unsigned int)sub_15A9520(v99, *(_DWORD *)(v13 + 8) >> 8);
        break;
    }
    break;
  }
  v28 = v17 * ((v17 + ((unsigned __int64)(v27 * v12 + 7) >> 3) - 1) / v17);
  v29 = *(_DWORD *)(v11 + 32);
  v30 = v28;
  if ( v29 > 0x40 )
  {
    if ( v29 - (unsigned int)sub_16A57B0(v11 + 24) > 0x40 )
    {
      v33 = 0;
      v31 = -1;
      goto LABEL_21;
    }
    v31 = **(_QWORD **)(v11 + 24);
  }
  else
  {
    v31 = *(_QWORD *)(v11 + 24);
  }
  if ( !v31 )
  {
LABEL_31:
    v38 = v31 * v30;
LABEL_32:
    v36 = v86 + v38;
    if ( v86 >= v38 )
      v38 = v86;
    if ( v36 < v38 )
      v36 = -1;
    goto LABEL_29;
  }
  _BitScanReverse64(&v32, v31);
  v33 = v32 ^ 0x3F;
LABEL_21:
  if ( !v30 )
    goto LABEL_31;
  _BitScanReverse64(&v34, v30);
  v35 = 126 - v33 - (v34 ^ 0x3F);
  if ( v35 <= 0x3E )
    goto LABEL_31;
  v36 = -1;
  if ( v35 != 63 )
    goto LABEL_29;
  v37 = v30 * (v31 >> 1);
  if ( v37 < 0 )
    goto LABEL_29;
  v38 = 2 * v37;
  if ( (v31 & 1) == 0 )
    goto LABEL_32;
  v39 = v30 + v38;
  if ( v30 >= v38 )
    v38 = v30;
  if ( v39 >= v38 )
  {
    v38 = v39;
    goto LABEL_32;
  }
LABEL_29:
  *(_QWORD *)(a1 + 104) = v36;
  v21 = *(_QWORD *)(a2 - 24);
  if ( (unsigned __int8)sub_3852EE0(a1, a2, a2, a1, v15, v16) )
    return 1;
LABEL_30:
  sub_384F350(a1, v21);
  return 0;
}
