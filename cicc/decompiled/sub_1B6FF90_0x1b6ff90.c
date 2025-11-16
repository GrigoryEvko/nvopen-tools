// Function: sub_1B6FF90
// Address: 0x1b6ff90
//
__int64 __fastcall sub_1B6FF90(__int64 ***a1, int a2, __int64 a3, _BYTE *a4, double a5, double a6, double a7)
{
  __int64 ***v7; // r14
  __int64 **v10; // rsi
  __int64 v11; // rax
  __int64 v12; // rbx
  unsigned int v13; // esi
  __int64 v14; // r10
  __int64 v15; // rsi
  __int64 v16; // rcx
  unsigned __int64 v17; // rbx
  __int64 v18; // r15
  __int64 **v19; // rsi
  char v20; // al
  unsigned __int64 v21; // r15
  unsigned int v22; // r13d
  __int64 **v23; // rax
  __int64 v25; // r15
  unsigned int v26; // eax
  __int64 v27; // rcx
  unsigned __int64 v28; // r10
  __int64 *v29; // rax
  __int64 v30; // rax
  int v31; // eax
  __int64 v32; // r15
  unsigned int v33; // eax
  __int64 v34; // rsi
  __int64 v35; // r10
  unsigned __int64 v36; // r11
  _QWORD *v37; // rax
  int v38; // eax
  __int64 **v39; // rax
  __int64 **v40; // rax
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // rax
  unsigned __int64 v44; // r15
  __int64 v45; // rax
  _QWORD *v46; // rax
  int v47; // eax
  __int64 v48; // rax
  _QWORD *v49; // rax
  unsigned int v50; // esi
  int v51; // eax
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 v54; // [rsp+0h] [rbp-70h]
  __int64 v55; // [rsp+8h] [rbp-68h]
  unsigned __int64 v56; // [rsp+8h] [rbp-68h]
  unsigned __int64 v57; // [rsp+10h] [rbp-60h]
  __int64 v58; // [rsp+10h] [rbp-60h]
  __int64 v59; // [rsp+18h] [rbp-58h]
  __int64 v60; // [rsp+18h] [rbp-58h]
  __int64 v61; // [rsp+18h] [rbp-58h]
  __int64 v62; // [rsp+18h] [rbp-58h]
  __int64 v63; // [rsp+18h] [rbp-58h]
  __int64 v64; // [rsp+18h] [rbp-58h]
  __int64 v65; // [rsp+20h] [rbp-50h]
  __int64 v66; // [rsp+20h] [rbp-50h]
  unsigned __int64 v67; // [rsp+20h] [rbp-50h]
  unsigned __int64 v68; // [rsp+20h] [rbp-50h]
  unsigned __int64 v69; // [rsp+20h] [rbp-50h]
  unsigned __int64 v70; // [rsp+20h] [rbp-50h]
  unsigned __int64 v71; // [rsp+20h] [rbp-50h]
  __int64 *v72; // [rsp+28h] [rbp-48h]
  __int64 v73; // [rsp+28h] [rbp-48h]
  __int64 v74; // [rsp+28h] [rbp-48h]
  __int64 v75; // [rsp+28h] [rbp-48h]
  __int64 v76; // [rsp+28h] [rbp-48h]
  __int64 v77; // [rsp+28h] [rbp-48h]
  __int64 v78; // [rsp+28h] [rbp-48h]
  __int64 v79; // [rsp+28h] [rbp-48h]
  __int64 *v80; // [rsp+30h] [rbp-40h]

  v7 = a1;
  v10 = *a1;
  v80 = **a1;
  v11 = *((unsigned __int8 *)*a1 + 8);
  if ( (_BYTE)v11 == 15 )
  {
    v12 = 1;
    v13 = *((_DWORD *)v10 + 2) >> 8;
    if ( *(_BYTE *)(a3 + 8) == 15 && *(_DWORD *)(a3 + 8) >> 8 == v13 )
      return sub_1B6E4B0((__int64)v7, a3, a4, a5, a6, a7);
LABEL_3:
    v14 = 8 * (unsigned int)sub_15A9520((__int64)a4, v13);
  }
  else
  {
    v12 = 1;
    while ( 2 )
    {
      switch ( v11 )
      {
        case 0LL:
        case 8LL:
        case 10LL:
        case 12LL:
        case 16LL:
          v29 = v10[4];
          v10 = (__int64 **)v10[3];
          v12 *= (_QWORD)v29;
          v11 = *((unsigned __int8 *)v10 + 8);
          continue;
        case 1LL:
          v14 = 16;
          break;
        case 2LL:
          v14 = 32;
          break;
        case 3LL:
        case 9LL:
          v14 = 64;
          break;
        case 4LL:
          v14 = 80;
          break;
        case 5LL:
        case 6LL:
          v14 = 128;
          break;
        case 7LL:
          v14 = 8 * (unsigned int)sub_15A9520((__int64)a4, 0);
          break;
        case 11LL:
          v14 = *((_DWORD *)v10 + 2) >> 8;
          break;
        case 13LL:
          v14 = 8LL * *(_QWORD *)sub_15A9930((__int64)a4, (__int64)v10);
          break;
        case 14LL:
          v25 = (__int64)v10[3];
          v72 = v10[4];
          v26 = sub_15A9FE0((__int64)a4, v25);
          v27 = 1;
          v28 = v26;
          while ( 2 )
          {
            switch ( *(_BYTE *)(v25 + 8) )
            {
              case 0:
              case 8:
              case 0xA:
              case 0xC:
              case 0x10:
                v48 = *(_QWORD *)(v25 + 32);
                v25 = *(_QWORD *)(v25 + 24);
                v27 *= v48;
                continue;
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
                JUMPOUT(0x1B704A0);
              case 0xB:
                JUMPOUT(0x1B70498);
              case 0xD:
                v60 = v27;
                v67 = v28;
                v46 = (_QWORD *)sub_15A9930((__int64)a4, v25);
                v28 = v67;
                v27 = v60;
                v42 = 8LL * *v46;
                goto LABEL_48;
              case 0xE:
                v55 = v27;
                v57 = v28;
                v59 = *(_QWORD *)(v25 + 24);
                v66 = *(_QWORD *)(v25 + 32);
                v44 = (unsigned int)sub_15A9FE0((__int64)a4, v59);
                v45 = sub_127FA20((__int64)a4, v59);
                v28 = v57;
                v27 = v55;
                v42 = 8 * v44 * v66 * ((v44 + ((unsigned __int64)(v45 + 7) >> 3) - 1) / v44);
                goto LABEL_48;
              case 0xF:
                v61 = v27;
                v68 = v28;
                v47 = sub_15A9520((__int64)a4, *(_DWORD *)(v25 + 8) >> 8);
                v28 = v68;
                v27 = v61;
                v42 = (unsigned int)(8 * v47);
LABEL_48:
                v14 = 8 * (_QWORD)v72 * v28 * ((v28 + ((unsigned __int64)(v42 * v27 + 7) >> 3) - 1) / v28);
                break;
            }
            break;
          }
          break;
        case 15LL:
          v13 = *((_DWORD *)v10 + 2) >> 8;
          goto LABEL_3;
      }
      break;
    }
  }
  v15 = a3;
  v16 = 1;
  v17 = (unsigned __int64)(v14 * v12 + 7) >> 3;
  while ( 2 )
  {
    switch ( *(_BYTE *)(v15 + 8) )
    {
      case 1:
        v18 = 16;
        goto LABEL_13;
      case 2:
        v18 = 32;
        goto LABEL_13;
      case 3:
      case 9:
        v18 = 64;
        goto LABEL_13;
      case 4:
        v18 = 80;
        goto LABEL_13;
      case 5:
      case 6:
        v18 = 128;
        goto LABEL_13;
      case 7:
        v76 = v16;
        v38 = sub_15A9520((__int64)a4, 0);
        v16 = v76;
        v18 = (unsigned int)(8 * v38);
        goto LABEL_13;
      case 0xB:
        v18 = *(_DWORD *)(v15 + 8) >> 8;
        goto LABEL_13;
      case 0xD:
        v75 = v16;
        v37 = (_QWORD *)sub_15A9930((__int64)a4, v15);
        v16 = v75;
        v18 = 8LL * *v37;
        goto LABEL_13;
      case 0xE:
        v32 = *(_QWORD *)(v15 + 32);
        v65 = v16;
        v74 = *(_QWORD *)(v15 + 24);
        v33 = sub_15A9FE0((__int64)a4, v74);
        v16 = v65;
        v34 = v74;
        v35 = 1;
        v36 = v33;
        while ( 2 )
        {
          switch ( *(_BYTE *)(v34 + 8) )
          {
            case 1:
              v43 = 16;
              goto LABEL_51;
            case 2:
              v43 = 32;
              goto LABEL_51;
            case 3:
            case 9:
              v43 = 64;
              goto LABEL_51;
            case 4:
              v43 = 80;
              goto LABEL_51;
            case 5:
            case 6:
              v43 = 128;
              goto LABEL_51;
            case 7:
              v63 = v35;
              v50 = 0;
              v70 = v36;
              v78 = v16;
              goto LABEL_64;
            case 0xB:
              v43 = *(_DWORD *)(v34 + 8) >> 8;
              goto LABEL_51;
            case 0xD:
              v62 = v35;
              v69 = v36;
              v77 = v16;
              v49 = (_QWORD *)sub_15A9930((__int64)a4, v34);
              v16 = v77;
              v36 = v69;
              v35 = v62;
              v43 = 8LL * *v49;
              goto LABEL_51;
            case 0xE:
              v54 = v35;
              v56 = v36;
              v58 = v65;
              v64 = *(_QWORD *)(v34 + 24);
              v79 = *(_QWORD *)(v34 + 32);
              v71 = (unsigned int)sub_15A9FE0((__int64)a4, v64);
              v53 = sub_127FA20((__int64)a4, v64);
              v16 = v58;
              v36 = v56;
              v35 = v54;
              v43 = 8 * v71 * v79 * ((v71 + ((unsigned __int64)(v53 + 7) >> 3) - 1) / v71);
              goto LABEL_51;
            case 0xF:
              v63 = v35;
              v70 = v36;
              v78 = v16;
              v50 = *(_DWORD *)(v34 + 8) >> 8;
LABEL_64:
              v51 = sub_15A9520((__int64)a4, v50);
              v16 = v78;
              v36 = v70;
              v35 = v63;
              v43 = (unsigned int)(8 * v51);
LABEL_51:
              v18 = 8 * v36 * v32 * ((v36 + ((unsigned __int64)(v43 * v35 + 7) >> 3) - 1) / v36);
              goto LABEL_13;
            case 0x10:
              v52 = *(_QWORD *)(v34 + 32);
              v34 = *(_QWORD *)(v34 + 24);
              v35 *= v52;
              continue;
            default:
              goto LABEL_71;
          }
        }
      case 0xF:
        v73 = v16;
        v31 = sub_15A9520((__int64)a4, *(_DWORD *)(v15 + 8) >> 8);
        v16 = v73;
        v18 = (unsigned int)(8 * v31);
LABEL_13:
        v19 = *a1;
        v20 = *((_BYTE *)*a1 + 8);
        v21 = (unsigned __int64)(v18 * v16 + 7) >> 3;
        if ( v20 == 16 )
        {
          if ( *(_BYTE *)(*v19[2] + 8) != 15 )
            goto LABEL_43;
        }
        else if ( v20 != 15 )
        {
          goto LABEL_15;
        }
        v40 = (__int64 **)sub_15A9650((__int64)a4, (__int64)v19);
        v7 = (__int64 ***)sub_15A46C0(45, a1, v40, 0);
        v20 = *((_BYTE *)*v7 + 8);
LABEL_15:
        if ( v20 == 11 )
          goto LABEL_16;
LABEL_43:
        v39 = (__int64 **)sub_1644900(v80, 8 * (int)v17);
        v7 = (__int64 ***)sub_15A46C0(47, v7, v39, 0);
LABEL_16:
        if ( *a4 )
          a2 = v17 - a2 - v21;
        v22 = 8 * a2;
        if ( v22 )
        {
          v41 = sub_15A0680((__int64)*v7, v22, 0);
          v7 = (__int64 ***)sub_15A2D80((__int64 *)v7, v41, 0, a5, a6, a7);
        }
        if ( v17 != v21 )
        {
          v23 = (__int64 **)sub_1644900(v80, 8 * (int)v21);
          v7 = (__int64 ***)sub_15A4670(v7, v23);
        }
        break;
      case 0x10:
        v30 = *(_QWORD *)(v15 + 32);
        v15 = *(_QWORD *)(v15 + 24);
        v16 *= v30;
        continue;
      default:
LABEL_71:
        BUG();
    }
    return sub_1B6E4B0((__int64)v7, a3, a4, a5, a6, a7);
  }
}
