// Function: sub_1B6F750
// Address: 0x1b6f750
//
__int64 __fastcall sub_1B6F750(__int64 a1, unsigned __int16 *a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rsi
  __int64 v7; // rax
  unsigned __int16 *v8; // r15
  __int64 v10; // rcx
  __int64 v11; // rax
  __int64 result; // rax
  _QWORD *v13; // rax
  unsigned int v14; // eax
  __int64 v15; // rsi
  __int64 v16; // r10
  unsigned __int64 v17; // r9
  int v18; // eax
  int v19; // eax
  __int64 v20; // rax
  unsigned __int16 *v21; // rax
  __int64 v22; // rsi
  __int64 v23; // rcx
  __int64 v24; // r9
  __int64 v25; // rdx
  int v26; // eax
  unsigned int v27; // esi
  int v28; // eax
  __int64 v29; // rax
  unsigned int v30; // eax
  __int64 v31; // r10
  __int64 v32; // rsi
  unsigned __int64 v33; // r11
  _QWORD *v34; // rax
  __int64 v35; // rax
  unsigned int v36; // esi
  int v37; // eax
  __int64 v38; // rax
  __int64 v39; // rax
  _QWORD *v40; // rax
  __int64 v41; // rax
  unsigned int v42; // esi
  int v43; // eax
  __int64 v44; // rax
  __int64 v45; // rax
  _QWORD *v46; // rax
  __int64 v47; // [rsp+0h] [rbp-80h]
  __int64 v48; // [rsp+8h] [rbp-78h]
  unsigned __int64 v49; // [rsp+8h] [rbp-78h]
  unsigned __int64 v50; // [rsp+10h] [rbp-70h]
  __int64 v51; // [rsp+10h] [rbp-70h]
  __int64 v52; // [rsp+18h] [rbp-68h]
  __int64 v53; // [rsp+18h] [rbp-68h]
  __int64 v54; // [rsp+18h] [rbp-68h]
  __int64 v55; // [rsp+18h] [rbp-68h]
  __int64 v56; // [rsp+20h] [rbp-60h]
  __int64 v57; // [rsp+20h] [rbp-60h]
  __int64 v58; // [rsp+20h] [rbp-60h]
  __int64 v59; // [rsp+20h] [rbp-60h]
  unsigned __int64 v60; // [rsp+20h] [rbp-60h]
  __int64 v61; // [rsp+20h] [rbp-60h]
  unsigned __int64 v62; // [rsp+20h] [rbp-60h]
  __int64 v63; // [rsp+28h] [rbp-58h]
  __int64 v64; // [rsp+28h] [rbp-58h]
  unsigned __int64 v65; // [rsp+28h] [rbp-58h]
  unsigned __int64 v66; // [rsp+28h] [rbp-58h]
  unsigned __int64 v67; // [rsp+28h] [rbp-58h]
  __int64 v68; // [rsp+28h] [rbp-58h]
  unsigned __int64 v69; // [rsp+28h] [rbp-58h]
  __int64 v70; // [rsp+28h] [rbp-58h]
  __int64 v71; // [rsp+30h] [rbp-50h]
  __int64 v72; // [rsp+30h] [rbp-50h]
  __int64 v73; // [rsp+30h] [rbp-50h]
  __int64 v74; // [rsp+30h] [rbp-50h]
  __int64 v75; // [rsp+30h] [rbp-50h]
  __int64 v76; // [rsp+30h] [rbp-50h]
  __int64 v77; // [rsp+30h] [rbp-50h]
  __int64 v78; // [rsp+30h] [rbp-50h]
  __int64 v79; // [rsp+30h] [rbp-50h]
  __int64 v80; // [rsp+30h] [rbp-50h]
  __int64 v81; // [rsp+38h] [rbp-48h]
  __int64 v82; // [rsp+38h] [rbp-48h]
  __int64 v83; // [rsp+38h] [rbp-48h]
  __int64 v84; // [rsp+38h] [rbp-48h]
  __int64 v85; // [rsp+38h] [rbp-48h]
  __int64 v86; // [rsp+38h] [rbp-48h]
  __int64 v87; // [rsp+38h] [rbp-48h]
  __int64 v88[7]; // [rsp+48h] [rbp-38h] BYREF

  v6 = *(_QWORD *)a3;
  v7 = *(unsigned __int8 *)(*(_QWORD *)a3 + 8LL);
  if ( (unsigned __int8)(v7 - 13) <= 1u )
    return 0xFFFFFFFFLL;
  v8 = *(unsigned __int16 **)(a3 - 24);
  v10 = 1;
  while ( 2 )
  {
    switch ( v7 )
    {
      case 0LL:
      case 8LL:
      case 10LL:
      case 12LL:
      case 16LL:
        v20 = *(_QWORD *)(v6 + 32);
        v6 = *(_QWORD *)(v6 + 24);
        v10 *= v20;
        v7 = *(unsigned __int8 *)(v6 + 8);
        continue;
      case 1LL:
        v11 = 16;
        break;
      case 2LL:
        v11 = 32;
        break;
      case 3LL:
      case 9LL:
        v11 = 64;
        break;
      case 4LL:
        v11 = 80;
        break;
      case 5LL:
      case 6LL:
        v11 = 128;
        break;
      case 7LL:
        v84 = v10;
        v19 = sub_15A9520(a4, 0);
        v10 = v84;
        v11 = (unsigned int)(8 * v19);
        break;
      case 11LL:
        v11 = *(_DWORD *)(v6 + 8) >> 8;
        break;
      case 13LL:
        v81 = v10;
        v13 = (_QWORD *)sub_15A9930(a4, v6);
        v10 = v81;
        v11 = 8LL * *v13;
        break;
      case 14LL:
        v63 = v10;
        v71 = *(_QWORD *)(v6 + 24);
        v82 = *(_QWORD *)(v6 + 32);
        v14 = sub_15A9FE0(a4, v71);
        v10 = v63;
        v15 = v71;
        v16 = 1;
        v17 = v14;
        while ( 2 )
        {
          switch ( *(_BYTE *)(v15 + 8) )
          {
            case 0:
            case 8:
            case 0xA:
            case 0xC:
            case 0x10:
              v38 = *(_QWORD *)(v15 + 32);
              v15 = *(_QWORD *)(v15 + 24);
              v16 *= v38;
              continue;
            case 1:
              v35 = 16;
              goto LABEL_36;
            case 2:
              v35 = 32;
              goto LABEL_36;
            case 3:
            case 9:
              v35 = 64;
              goto LABEL_36;
            case 4:
              v35 = 80;
              goto LABEL_36;
            case 5:
            case 6:
              v35 = 128;
              goto LABEL_36;
            case 7:
              v57 = v16;
              v36 = 0;
              v65 = v17;
              v75 = v10;
              goto LABEL_40;
            case 0xB:
              v35 = *(_DWORD *)(v15 + 8) >> 8;
              goto LABEL_36;
            case 0xD:
              v59 = v16;
              v67 = v17;
              v77 = v10;
              v40 = (_QWORD *)sub_15A9930(a4, v15);
              v10 = v77;
              v17 = v67;
              v16 = v59;
              v35 = 8LL * *v40;
              goto LABEL_36;
            case 0xE:
              v48 = v16;
              v50 = v17;
              v52 = v63;
              v58 = *(_QWORD *)(v15 + 24);
              v76 = *(_QWORD *)(v15 + 32);
              v66 = (unsigned int)sub_15A9FE0(a4, v58);
              v39 = sub_127FA20(a4, v58);
              v10 = v52;
              v17 = v50;
              v16 = v48;
              v35 = 8 * v76 * v66 * ((v66 + ((unsigned __int64)(v39 + 7) >> 3) - 1) / v66);
              goto LABEL_36;
            case 0xF:
              v57 = v16;
              v65 = v17;
              v75 = v10;
              v36 = *(_DWORD *)(v15 + 8) >> 8;
LABEL_40:
              v37 = sub_15A9520(a4, v36);
              v10 = v75;
              v17 = v65;
              v16 = v57;
              v35 = (unsigned int)(8 * v37);
LABEL_36:
              v11 = 8 * v17 * v82 * ((v17 + ((unsigned __int64)(v35 * v16 + 7) >> 3) - 1) / v17);
              break;
          }
          break;
        }
        break;
      case 15LL:
        v83 = v10;
        v18 = sub_15A9520(a4, *(_DWORD *)(v6 + 8) >> 8);
        v10 = v83;
        v11 = (unsigned int)(8 * v18);
        break;
    }
    break;
  }
  result = sub_1B6E190(a1, a2, v8, v11 * v10, a4);
  if ( (_DWORD)result == -1 )
  {
    v88[0] = 0;
    v21 = sub_14AC610(a2, v88, a4);
    v22 = a1;
    v23 = 1;
    v24 = (__int64)v21;
    while ( 2 )
    {
      switch ( *(_BYTE *)(v22 + 8) )
      {
        case 1:
          v25 = 16;
          goto LABEL_21;
        case 2:
          v25 = 32;
          goto LABEL_21;
        case 3:
        case 9:
          v25 = 64;
          goto LABEL_21;
        case 4:
          v25 = 80;
          goto LABEL_21;
        case 5:
        case 6:
          v25 = 128;
          goto LABEL_21;
        case 7:
          v72 = v23;
          v27 = 0;
          v85 = v24;
          goto LABEL_28;
        case 0xB:
          v25 = *(_DWORD *)(v22 + 8) >> 8;
          goto LABEL_21;
        case 0xD:
          v74 = v23;
          v87 = v24;
          v34 = (_QWORD *)sub_15A9930(a4, v22);
          v24 = v87;
          v23 = v74;
          v25 = 8LL * *v34;
          goto LABEL_21;
        case 0xE:
          v56 = v23;
          v64 = v24;
          v73 = *(_QWORD *)(v22 + 24);
          v86 = *(_QWORD *)(v22 + 32);
          v30 = sub_15A9FE0(a4, v73);
          v23 = v56;
          v24 = v64;
          v31 = 1;
          v32 = v73;
          v33 = v30;
          while ( 2 )
          {
            switch ( *(_BYTE *)(v32 + 8) )
            {
              case 1:
                v41 = 16;
                goto LABEL_49;
              case 2:
                v41 = 32;
                goto LABEL_49;
              case 3:
              case 9:
                v41 = 64;
                goto LABEL_49;
              case 4:
                v41 = 80;
                goto LABEL_49;
              case 5:
              case 6:
                v41 = 128;
                goto LABEL_49;
              case 7:
                v53 = v31;
                v42 = 0;
                v60 = v33;
                v68 = v23;
                v78 = v24;
                goto LABEL_52;
              case 0xB:
                v41 = *(_DWORD *)(v32 + 8) >> 8;
                goto LABEL_49;
              case 0xD:
                v55 = v31;
                v62 = v33;
                v70 = v23;
                v80 = v24;
                v46 = (_QWORD *)sub_15A9930(a4, v32);
                v24 = v80;
                v23 = v70;
                v33 = v62;
                v31 = v55;
                v41 = 8LL * *v46;
                goto LABEL_49;
              case 0xE:
                v47 = v31;
                v49 = v33;
                v51 = v56;
                v54 = v64;
                v61 = *(_QWORD *)(v32 + 24);
                v79 = *(_QWORD *)(v32 + 32);
                v69 = (unsigned int)sub_15A9FE0(a4, v61);
                v45 = sub_127FA20(a4, v61);
                v24 = v54;
                v23 = v51;
                v33 = v49;
                v31 = v47;
                v41 = 8 * v69 * v79 * ((v69 + ((unsigned __int64)(v45 + 7) >> 3) - 1) / v69);
                goto LABEL_49;
              case 0xF:
                v53 = v31;
                v60 = v33;
                v68 = v23;
                v42 = *(_DWORD *)(v32 + 8) >> 8;
                v78 = v24;
LABEL_52:
                v43 = sub_15A9520(a4, v42);
                v24 = v78;
                v23 = v68;
                v33 = v60;
                v31 = v53;
                v41 = (unsigned int)(8 * v43);
LABEL_49:
                v25 = 8 * v33 * v86 * ((v33 + ((unsigned __int64)(v41 * v31 + 7) >> 3) - 1) / v33);
                goto LABEL_21;
              case 0x10:
                v44 = *(_QWORD *)(v32 + 32);
                v32 = *(_QWORD *)(v32 + 24);
                v31 *= v44;
                continue;
              default:
                goto LABEL_62;
            }
          }
        case 0xF:
          v72 = v23;
          v85 = v24;
          v27 = *(_DWORD *)(v22 + 8) >> 8;
LABEL_28:
          v28 = sub_15A9520(a4, v27);
          v24 = v85;
          v23 = v72;
          v25 = (unsigned int)(8 * v28);
LABEL_21:
          v26 = sub_1412DC0(v24, v88[0], (unsigned __int64)(v25 * v23 + 7) >> 3, a3);
          if ( !v26 )
            return 0xFFFFFFFFLL;
          result = sub_1B6E190(a1, a2, v8, (unsigned int)(8 * v26), a4);
          break;
        case 0x10:
          v29 = *(_QWORD *)(v22 + 32);
          v22 = *(_QWORD *)(v22 + 24);
          v23 *= v29;
          continue;
        default:
LABEL_62:
          BUG();
      }
      break;
    }
  }
  return result;
}
