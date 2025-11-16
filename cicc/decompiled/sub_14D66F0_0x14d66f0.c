// Function: sub_14D66F0
// Address: 0x14d66f0
//
__int64 __fastcall sub_14D66F0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r13
  __int64 v5; // rsi
  __int64 v6; // rbx
  __int64 v7; // rax
  __int64 v8; // rbx
  __int64 v9; // rsi
  __int64 v10; // rcx
  __int64 v11; // rax
  char v13; // al
  unsigned int v14; // ebx
  __int64 v15; // rax
  unsigned int v16; // eax
  __int64 v17; // rsi
  __int64 v18; // rcx
  unsigned __int64 v19; // r10
  __int64 v20; // rax
  int v21; // eax
  unsigned int v22; // eax
  __int64 v23; // rsi
  __int64 v24; // r10
  unsigned __int64 v25; // r11
  _QWORD *v26; // rax
  int v27; // eax
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rsi
  int v31; // eax
  _QWORD *v32; // rax
  unsigned int v33; // eax
  __int64 v34; // rsi
  __int64 v35; // r8
  unsigned __int64 v36; // r11
  __int64 v37; // rax
  _QWORD *v38; // rax
  __int64 v39; // rsi
  int v40; // eax
  __int64 v41; // rax
  unsigned int v42; // eax
  __int64 v43; // r9
  __int64 v44; // rsi
  unsigned __int64 v45; // r8
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // rsi
  int v49; // eax
  __int64 v50; // rax
  _QWORD *v51; // rax
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 v54; // rsi
  __int64 v55; // rax
  __int64 v56; // [rsp+0h] [rbp-90h]
  __int64 v57; // [rsp+8h] [rbp-88h]
  __int64 v58; // [rsp+8h] [rbp-88h]
  __int64 v59; // [rsp+10h] [rbp-80h]
  __int64 v60; // [rsp+10h] [rbp-80h]
  unsigned __int64 v61; // [rsp+18h] [rbp-78h]
  __int64 v62; // [rsp+18h] [rbp-78h]
  unsigned __int64 v63; // [rsp+18h] [rbp-78h]
  __int64 v64; // [rsp+20h] [rbp-70h]
  __int64 v65; // [rsp+20h] [rbp-70h]
  __int64 v66; // [rsp+20h] [rbp-70h]
  __int64 v67; // [rsp+20h] [rbp-70h]
  __int64 v68; // [rsp+20h] [rbp-70h]
  __int64 v69; // [rsp+20h] [rbp-70h]
  __int64 v70; // [rsp+28h] [rbp-68h]
  __int64 v71; // [rsp+28h] [rbp-68h]
  __int64 v72; // [rsp+28h] [rbp-68h]
  __int64 v73; // [rsp+28h] [rbp-68h]
  __int64 v74; // [rsp+28h] [rbp-68h]
  __int64 v75; // [rsp+28h] [rbp-68h]
  __int64 v76; // [rsp+28h] [rbp-68h]
  __int64 v77; // [rsp+30h] [rbp-60h]
  __int64 v78; // [rsp+30h] [rbp-60h]
  __int64 v79; // [rsp+30h] [rbp-60h]
  __int64 v80; // [rsp+30h] [rbp-60h]
  unsigned __int64 v81; // [rsp+30h] [rbp-60h]
  unsigned __int64 v82; // [rsp+30h] [rbp-60h]
  unsigned __int64 v83; // [rsp+30h] [rbp-60h]
  unsigned __int64 v84; // [rsp+30h] [rbp-60h]
  unsigned __int64 v85; // [rsp+30h] [rbp-60h]
  __int64 v86; // [rsp+38h] [rbp-58h]
  unsigned __int64 v87; // [rsp+38h] [rbp-58h]
  unsigned __int64 v88; // [rsp+38h] [rbp-58h]
  __int64 v89; // [rsp+38h] [rbp-58h]
  unsigned __int64 v90; // [rsp+38h] [rbp-58h]
  unsigned __int64 v91; // [rsp+38h] [rbp-58h]
  __int64 v92; // [rsp+38h] [rbp-58h]
  __int64 v93; // [rsp+38h] [rbp-58h]
  __int64 v94; // [rsp+38h] [rbp-58h]
  __int64 v95; // [rsp+38h] [rbp-58h]
  __int64 v96; // [rsp+38h] [rbp-58h]
  __int64 v97; // [rsp+38h] [rbp-58h]
  __int64 v98; // [rsp+40h] [rbp-50h]
  __int64 v99; // [rsp+40h] [rbp-50h]
  __int64 v100; // [rsp+40h] [rbp-50h]
  __int64 v101; // [rsp+40h] [rbp-50h]
  __int64 v102; // [rsp+40h] [rbp-50h]
  __int64 v103; // [rsp+40h] [rbp-50h]
  __int64 v104; // [rsp+40h] [rbp-50h]
  __int64 v105; // [rsp+40h] [rbp-50h]
  __int64 v106; // [rsp+48h] [rbp-48h]
  __int64 v107; // [rsp+48h] [rbp-48h]
  __int64 v108; // [rsp+48h] [rbp-48h]
  __int64 v109; // [rsp+48h] [rbp-48h]
  __int64 v110; // [rsp+48h] [rbp-48h]

  while ( 2 )
  {
    v4 = *a1;
    v5 = a2;
    v6 = 1;
    while ( 2 )
    {
      switch ( *(_BYTE *)(v5 + 8) )
      {
        case 0:
        case 8:
        case 0xA:
        case 0xC:
        case 0x10:
          v15 = *(_QWORD *)(v5 + 32);
          v5 = *(_QWORD *)(v5 + 24);
          v6 *= v15;
          continue;
        case 1:
          v7 = 16;
          break;
        case 2:
          v7 = 32;
          break;
        case 3:
        case 9:
          v7 = 64;
          break;
        case 4:
          v7 = 80;
          break;
        case 5:
        case 6:
          v7 = 128;
          break;
        case 7:
          v7 = 8 * (unsigned int)sub_15A9520(a3, 0);
          break;
        case 0xB:
          v7 = *(_DWORD *)(v5 + 8) >> 8;
          break;
        case 0xD:
          v7 = 8LL * *(_QWORD *)sub_15A9930(a3, v5);
          break;
        case 0xE:
          v106 = *(_QWORD *)(v5 + 32);
          v98 = *(_QWORD *)(v5 + 24);
          v16 = sub_15A9FE0(a3, v98);
          v17 = v98;
          v18 = 1;
          v19 = v16;
          while ( 2 )
          {
            switch ( *(_BYTE *)(v17 + 8) )
            {
              case 0:
              case 8:
              case 0xA:
              case 0xC:
              case 0x10:
                v37 = *(_QWORD *)(v17 + 32);
                v17 = *(_QWORD *)(v17 + 24);
                v18 *= v37;
                continue;
              case 1:
                v28 = 16;
                goto LABEL_40;
              case 2:
                v28 = 32;
                goto LABEL_40;
              case 3:
              case 9:
                v28 = 64;
                goto LABEL_40;
              case 4:
                v28 = 80;
                goto LABEL_40;
              case 5:
              case 6:
                v28 = 128;
                goto LABEL_40;
              case 7:
                v87 = v19;
                v30 = 0;
                v100 = v18;
                goto LABEL_46;
              case 0xB:
                v28 = *(_DWORD *)(v17 + 8) >> 8;
                goto LABEL_40;
              case 0xD:
                v88 = v19;
                v101 = v18;
                v32 = (_QWORD *)sub_15A9930(a3, v17);
                v18 = v101;
                v19 = v88;
                v28 = 8LL * *v32;
                goto LABEL_40;
              case 0xE:
                v70 = v19;
                v77 = v18;
                v102 = *(_QWORD *)(v17 + 32);
                v89 = *(_QWORD *)(v17 + 24);
                v33 = sub_15A9FE0(a3, v89);
                v19 = v70;
                v34 = v89;
                v35 = 1;
                v18 = v77;
                v36 = v33;
                while ( 2 )
                {
                  switch ( *(_BYTE *)(v34 + 8) )
                  {
                    case 0:
                    case 8:
                    case 0xA:
                    case 0xC:
                    case 0x10:
                      v52 = *(_QWORD *)(v34 + 32);
                      v34 = *(_QWORD *)(v34 + 24);
                      v35 *= v52;
                      continue;
                    case 1:
                      v46 = 16;
                      goto LABEL_68;
                    case 2:
                      v46 = 32;
                      goto LABEL_68;
                    case 3:
                    case 9:
                      v46 = 64;
                      goto LABEL_68;
                    case 4:
                      v46 = 80;
                      goto LABEL_68;
                    case 5:
                    case 6:
                      v46 = 128;
                      goto LABEL_68;
                    case 7:
                      v65 = v70;
                      v48 = 0;
                      v72 = v77;
                      v81 = v36;
                      v93 = v35;
                      goto LABEL_75;
                    case 0xB:
                      v46 = *(_DWORD *)(v34 + 8) >> 8;
                      goto LABEL_68;
                    case 0xD:
                      v67 = v70;
                      v74 = v77;
                      v83 = v36;
                      v95 = v35;
                      v51 = (_QWORD *)sub_15A9930(a3, v34);
                      v35 = v95;
                      v36 = v83;
                      v18 = v74;
                      v19 = v67;
                      v46 = 8LL * *v51;
                      goto LABEL_68;
                    case 0xE:
                      v57 = v70;
                      v59 = v77;
                      v61 = v36;
                      v66 = v35;
                      v73 = *(_QWORD *)(v34 + 24);
                      v94 = *(_QWORD *)(v34 + 32);
                      v82 = (unsigned int)sub_15A9FE0(a3, v73);
                      v50 = sub_127FA20(a3, v73);
                      v35 = v66;
                      v36 = v61;
                      v18 = v59;
                      v19 = v57;
                      v46 = 8 * v82 * v94 * ((v82 + ((unsigned __int64)(v50 + 7) >> 3) - 1) / v82);
                      goto LABEL_68;
                    case 0xF:
                      v65 = v70;
                      v72 = v77;
                      v81 = v36;
                      v48 = *(_DWORD *)(v34 + 8) >> 8;
                      v93 = v35;
LABEL_75:
                      v49 = sub_15A9520(a3, v48);
                      v35 = v93;
                      v36 = v81;
                      v18 = v72;
                      v19 = v65;
                      v46 = (unsigned int)(8 * v49);
LABEL_68:
                      v28 = 8 * v102 * v36 * ((v36 + ((unsigned __int64)(v35 * v46 + 7) >> 3) - 1) / v36);
                      break;
                  }
                  goto LABEL_40;
                }
              case 0xF:
                v87 = v19;
                v100 = v18;
                v30 = *(_DWORD *)(v17 + 8) >> 8;
LABEL_46:
                v31 = sub_15A9520(a3, v30);
                v18 = v100;
                v19 = v87;
                v28 = (unsigned int)(8 * v31);
LABEL_40:
                v7 = 8 * v106 * v19 * ((v19 + ((unsigned __int64)(v28 * v18 + 7) >> 3) - 1) / v19);
                break;
            }
            break;
          }
          break;
        case 0xF:
          v7 = 8 * (unsigned int)sub_15A9520(a3, *(_DWORD *)(v5 + 8) >> 8);
          break;
      }
      break;
    }
    v8 = v7 * v6;
    v9 = v4;
    v10 = 1;
LABEL_5:
    switch ( *(_BYTE *)(v9 + 8) )
    {
      case 1:
        v11 = 16;
        goto LABEL_8;
      case 2:
        v11 = 32;
        goto LABEL_8;
      case 3:
      case 9:
        v11 = 64;
        goto LABEL_8;
      case 4:
        v11 = 80;
        goto LABEL_8;
      case 5:
      case 6:
        if ( v10 << 7 == v8 )
          goto LABEL_13;
        goto LABEL_9;
      case 7:
        v110 = v10;
        v27 = sub_15A9520(a3, 0);
        v10 = v110;
        v11 = (unsigned int)(8 * v27);
        goto LABEL_8;
      case 0xB:
        v11 = *(_DWORD *)(v9 + 8) >> 8;
        goto LABEL_8;
      case 0xD:
        v109 = v10;
        v26 = (_QWORD *)sub_15A9930(a3, v9);
        v10 = v109;
        v11 = 8LL * *v26;
        goto LABEL_8;
      case 0xE:
        v86 = v10;
        v108 = *(_QWORD *)(v9 + 32);
        v99 = *(_QWORD *)(v9 + 24);
        v22 = sub_15A9FE0(a3, v99);
        v23 = v99;
        v10 = v86;
        v24 = 1;
        v25 = v22;
        while ( 2 )
        {
          switch ( *(_BYTE *)(v23 + 8) )
          {
            case 1:
              v29 = 16;
              goto LABEL_43;
            case 2:
              v29 = 32;
              goto LABEL_43;
            case 3:
            case 9:
              v29 = 64;
              goto LABEL_43;
            case 4:
              v29 = 80;
              goto LABEL_43;
            case 5:
            case 6:
              v29 = 128;
              goto LABEL_43;
            case 7:
              v79 = v86;
              v39 = 0;
              v91 = v25;
              v104 = v24;
              goto LABEL_59;
            case 0xB:
              v29 = *(_DWORD *)(v23 + 8) >> 8;
              goto LABEL_43;
            case 0xD:
              v78 = v86;
              v90 = v25;
              v103 = v24;
              v38 = (_QWORD *)sub_15A9930(a3, v23);
              v24 = v103;
              v25 = v90;
              v10 = v78;
              v29 = 8LL * *v38;
              goto LABEL_43;
            case 0xE:
              v64 = v86;
              v71 = v25;
              v80 = v24;
              v92 = *(_QWORD *)(v23 + 24);
              v105 = *(_QWORD *)(v23 + 32);
              v42 = sub_15A9FE0(a3, v92);
              v10 = v64;
              v25 = v71;
              v43 = 1;
              v44 = v92;
              v24 = v80;
              v45 = v42;
              while ( 2 )
              {
                switch ( *(_BYTE *)(v44 + 8) )
                {
                  case 1:
                    v47 = 16;
                    goto LABEL_71;
                  case 2:
                    v47 = 32;
                    goto LABEL_71;
                  case 3:
                  case 9:
                    v47 = 64;
                    goto LABEL_71;
                  case 4:
                    v47 = 80;
                    goto LABEL_71;
                  case 5:
                  case 6:
                    v47 = 128;
                    goto LABEL_71;
                  case 7:
                    v62 = v64;
                    v54 = 0;
                    v68 = v71;
                    v75 = v80;
                    v84 = v45;
                    v96 = v43;
                    goto LABEL_85;
                  case 0xB:
                    v47 = *(_DWORD *)(v44 + 8) >> 8;
                    goto LABEL_71;
                  case 0xD:
                    v62 = v64;
                    v68 = v71;
                    v75 = v80;
                    v84 = v45;
                    v96 = v43;
                    v47 = 8LL * *(_QWORD *)sub_15A9930(a3, v44);
                    goto LABEL_86;
                  case 0xE:
                    v56 = v64;
                    v58 = v71;
                    v60 = v80;
                    v63 = v45;
                    v69 = v43;
                    v97 = *(_QWORD *)(v44 + 32);
                    v76 = *(_QWORD *)(v44 + 24);
                    v85 = (unsigned int)sub_15A9FE0(a3, v76);
                    v55 = sub_127FA20(a3, v76);
                    v43 = v69;
                    v45 = v63;
                    v24 = v60;
                    v25 = v58;
                    v10 = v56;
                    v47 = 8 * v85 * v97 * ((v85 + ((unsigned __int64)(v55 + 7) >> 3) - 1) / v85);
                    goto LABEL_71;
                  case 0xF:
                    v62 = v64;
                    v68 = v71;
                    v75 = v80;
                    v54 = *(_DWORD *)(v44 + 8) >> 8;
                    v84 = v45;
                    v96 = v43;
LABEL_85:
                    v47 = 8 * (unsigned int)sub_15A9520(a3, v54);
LABEL_86:
                    v43 = v96;
                    v45 = v84;
                    v24 = v75;
                    v25 = v68;
                    v10 = v62;
LABEL_71:
                    v29 = 8 * v45 * v105 * ((v45 + ((unsigned __int64)(v43 * v47 + 7) >> 3) - 1) / v45);
                    goto LABEL_43;
                  case 0x10:
                    v53 = *(_QWORD *)(v44 + 32);
                    v44 = *(_QWORD *)(v44 + 24);
                    v43 *= v53;
                    continue;
                  default:
                    goto LABEL_94;
                }
              }
            case 0xF:
              v79 = v86;
              v91 = v25;
              v104 = v24;
              v39 = *(_DWORD *)(v23 + 8) >> 8;
LABEL_59:
              v40 = sub_15A9520(a3, v39);
              v24 = v104;
              v25 = v91;
              v10 = v79;
              v29 = (unsigned int)(8 * v40);
LABEL_43:
              v11 = 8 * v108 * v25 * ((v25 + ((unsigned __int64)(v29 * v24 + 7) >> 3) - 1) / v25);
              goto LABEL_8;
            case 0x10:
              v41 = *(_QWORD *)(v23 + 32);
              v23 = *(_QWORD *)(v23 + 24);
              v24 *= v41;
              continue;
            default:
              goto LABEL_94;
          }
        }
      case 0xF:
        v107 = v10;
        v21 = sub_15A9520(a3, *(_DWORD *)(v9 + 8) >> 8);
        v10 = v107;
        v11 = (unsigned int)(8 * v21);
LABEL_8:
        if ( v11 * v10 != v8 )
          goto LABEL_9;
LABEL_13:
        v13 = *(_BYTE *)(v4 + 8);
        if ( v13 == 11 )
        {
          v14 = (*(_BYTE *)(a2 + 8) != 15) + 46;
        }
        else
        {
          v14 = 47;
          if ( v13 == 15 )
            v14 = 2 * (*(_BYTE *)(a2 + 8) != 11) + 45;
        }
        if ( !(unsigned __int8)sub_15FC090(v14, a1, a2) )
        {
LABEL_9:
          if ( (unsigned int)*(unsigned __int8 *)(v4 + 8) - 13 > 1 )
            return 0;
          a1 = (__int64 *)sub_15A0A60(a1, 0);
          if ( !a1 )
            return 0;
          continue;
        }
        return sub_15A46C0(v14, a1, a2, 0);
      case 0x10:
        v20 = *(_QWORD *)(v9 + 32);
        v9 = *(_QWORD *)(v9 + 24);
        v10 *= v20;
        goto LABEL_5;
      default:
LABEL_94:
        JUMPOUT(0x419798);
    }
  }
}
