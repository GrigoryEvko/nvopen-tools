// Function: sub_1B725B0
// Address: 0x1b725b0
//
__int64 __fastcall sub_1B725B0(__int64 ***a1, int a2, __int64 a3, __int64 a4, _BYTE *a5)
{
  __int64 ***v5; // r15
  __int64 v9; // rax
  unsigned __int8 *v10; // rsi
  __int64 v11; // rax
  __int64 **v12; // rsi
  __int64 v13; // r12
  __int64 v14; // rax
  unsigned int v15; // esi
  __int64 v16; // rbx
  __int64 v17; // rsi
  __int64 v18; // rbx
  __int64 *v19; // rax
  __int64 v20; // r12
  __int64 **v21; // rsi
  char v22; // al
  unsigned __int64 v23; // r12
  __int64 v24; // rsi
  __int64 **v25; // rsi
  __int64 v26; // r12
  unsigned int v28; // eax
  __int64 v29; // rsi
  __int64 v30; // r10
  unsigned __int64 v31; // rbx
  __int64 v32; // rax
  __int64 v33; // r12
  unsigned int v34; // eax
  __int64 v35; // rsi
  __int64 v36; // r10
  unsigned __int64 v37; // r8
  __int64 v38; // rax
  __int64 **v39; // rdx
  __int64 **v40; // rdx
  __int64 v41; // rax
  unsigned __int64 *v42; // rbx
  __int64 **v43; // rax
  unsigned __int64 v44; // rsi
  __int64 v45; // rsi
  unsigned __int8 *v46; // rsi
  __int64 v47; // rax
  unsigned __int64 *v48; // r12
  __int64 **v49; // rax
  unsigned __int64 v50; // rcx
  __int64 v51; // rsi
  unsigned __int8 *v52; // rsi
  __int64 v53; // rax
  __int64 v54; // rax
  __int64 v55; // rax
  int v56; // eax
  int v57; // eax
  __int64 v58; // rax
  _QWORD *v59; // rax
  __int64 v60; // rax
  unsigned int v61; // esi
  int v62; // eax
  __int64 v63; // rax
  _QWORD *v64; // rax
  __int64 v65; // rax
  unsigned __int64 *v66; // rbx
  __int64 **v67; // rax
  unsigned __int64 v68; // rsi
  __int64 v69; // rsi
  unsigned __int8 *v70; // rsi
  __int64 v71; // [rsp+0h] [rbp-110h]
  __int64 v72; // [rsp+8h] [rbp-108h]
  unsigned __int64 v73; // [rsp+8h] [rbp-108h]
  __int64 v74; // [rsp+10h] [rbp-100h]
  __int64 v75; // [rsp+10h] [rbp-100h]
  unsigned __int64 v76; // [rsp+18h] [rbp-F8h]
  __int64 v77; // [rsp+18h] [rbp-F8h]
  unsigned __int64 v78; // [rsp+18h] [rbp-F8h]
  __int64 v79; // [rsp+18h] [rbp-F8h]
  __int64 v80; // [rsp+20h] [rbp-F0h]
  __int64 v81; // [rsp+20h] [rbp-F0h]
  __int64 v82; // [rsp+20h] [rbp-F0h]
  __int64 v83; // [rsp+20h] [rbp-F0h]
  __int64 v84; // [rsp+20h] [rbp-F0h]
  __int64 v85; // [rsp+20h] [rbp-F0h]
  unsigned __int64 v86; // [rsp+20h] [rbp-F0h]
  __int64 v87; // [rsp+20h] [rbp-F0h]
  unsigned __int64 v88; // [rsp+20h] [rbp-F0h]
  __int64 *v89; // [rsp+28h] [rbp-E8h]
  unsigned __int64 v91; // [rsp+38h] [rbp-D8h]
  __int64 *v92; // [rsp+38h] [rbp-D8h]
  unsigned __int8 *v93; // [rsp+48h] [rbp-C8h] BYREF
  __int64 v94[2]; // [rsp+50h] [rbp-C0h] BYREF
  __int16 v95; // [rsp+60h] [rbp-B0h]
  unsigned __int8 *v96[2]; // [rsp+70h] [rbp-A0h] BYREF
  __int16 v97; // [rsp+80h] [rbp-90h]
  unsigned __int8 *v98; // [rsp+90h] [rbp-80h] BYREF
  __int64 v99; // [rsp+98h] [rbp-78h]
  unsigned __int64 *v100; // [rsp+A0h] [rbp-70h]
  __int64 v101; // [rsp+A8h] [rbp-68h]
  __int64 v102; // [rsp+B0h] [rbp-60h]
  int v103; // [rsp+B8h] [rbp-58h]
  __int64 v104; // [rsp+C0h] [rbp-50h]
  __int64 v105; // [rsp+C8h] [rbp-48h]

  v5 = a1;
  v9 = sub_16498A0(a4);
  v10 = *(unsigned __int8 **)(a4 + 48);
  v98 = 0;
  v101 = v9;
  v11 = *(_QWORD *)(a4 + 40);
  v102 = 0;
  v99 = v11;
  v103 = 0;
  v104 = 0;
  v105 = 0;
  v100 = (unsigned __int64 *)(a4 + 24);
  v96[0] = v10;
  if ( v10 )
  {
    sub_1623A60((__int64)v96, (__int64)v10, 2);
    if ( v98 )
      sub_161E7C0((__int64)&v98, (__int64)v98);
    v98 = v96[0];
    if ( v96[0] )
      sub_1623210((__int64)v96, v96[0], (__int64)&v98);
  }
  v12 = *a1;
  v13 = 1;
  v89 = **a1;
  v14 = *((unsigned __int8 *)*a1 + 8);
  if ( (_BYTE)v14 == 15 )
  {
    v13 = 1;
    v15 = *((_DWORD *)v12 + 2) >> 8;
    if ( *(_BYTE *)(a3 + 8) == 15 && *(_DWORD *)(a3 + 8) >> 8 == v15 )
      goto LABEL_27;
LABEL_8:
    v16 = 8 * (unsigned int)sub_15A9520((__int64)a5, v15);
  }
  else
  {
    while ( 2 )
    {
      switch ( v14 )
      {
        case 0LL:
        case 8LL:
        case 10LL:
        case 12LL:
        case 16LL:
          v19 = v12[4];
          v12 = (__int64 **)v12[3];
          v13 *= (_QWORD)v19;
          v14 = *((unsigned __int8 *)v12 + 8);
          continue;
        case 1LL:
          v16 = 16;
          break;
        case 2LL:
          v16 = 32;
          break;
        case 3LL:
        case 9LL:
          v16 = 64;
          break;
        case 4LL:
          v16 = 80;
          break;
        case 5LL:
        case 6LL:
          v16 = 128;
          break;
        case 7LL:
          v16 = 8 * (unsigned int)sub_15A9520((__int64)a5, 0);
          break;
        case 11LL:
          v16 = *((_DWORD *)v12 + 2) >> 8;
          break;
        case 13LL:
          v16 = 8LL * *(_QWORD *)sub_15A9930((__int64)a5, (__int64)v12);
          break;
        case 14LL:
          v80 = (__int64)v12[3];
          v92 = v12[4];
          v28 = sub_15A9FE0((__int64)a5, v80);
          v29 = v80;
          v30 = 1;
          v31 = v28;
          while ( 2 )
          {
            switch ( *(_BYTE *)(v29 + 8) )
            {
              case 0:
              case 8:
              case 0xA:
              case 0xC:
              case 0x10:
                v55 = *(_QWORD *)(v29 + 32);
                v29 = *(_QWORD *)(v29 + 24);
                v30 *= v55;
                continue;
              case 1:
                v53 = 16;
                break;
              case 2:
                v53 = 32;
                break;
              case 3:
              case 9:
                v53 = 64;
                break;
              case 4:
                v53 = 80;
                break;
              case 5:
              case 6:
                v53 = 128;
                break;
              case 7:
                v82 = v30;
                v56 = sub_15A9520((__int64)a5, 0);
                v30 = v82;
                v53 = (unsigned int)(8 * v56);
                break;
              case 0xB:
                v53 = *(_DWORD *)(v29 + 8) >> 8;
                break;
              case 0xD:
                v85 = v30;
                v59 = (_QWORD *)sub_15A9930((__int64)a5, v29);
                v30 = v85;
                v53 = 8LL * *v59;
                break;
              case 0xE:
                v72 = v30;
                v74 = *(_QWORD *)(v29 + 24);
                v84 = *(_QWORD *)(v29 + 32);
                v76 = (unsigned int)sub_15A9FE0((__int64)a5, v74);
                v58 = sub_127FA20((__int64)a5, v74);
                v30 = v72;
                v53 = 8 * v84 * v76 * ((v76 + ((unsigned __int64)(v58 + 7) >> 3) - 1) / v76);
                break;
              case 0xF:
                v83 = v30;
                v57 = sub_15A9520((__int64)a5, *(_DWORD *)(v29 + 8) >> 8);
                v30 = v83;
                v53 = (unsigned int)(8 * v57);
                break;
            }
            break;
          }
          v16 = 8 * (_QWORD)v92 * v31 * ((v31 + ((unsigned __int64)(v53 * v30 + 7) >> 3) - 1) / v31);
          break;
        case 15LL:
          v15 = *((_DWORD *)v12 + 2) >> 8;
          goto LABEL_8;
      }
      break;
    }
  }
  v17 = a3;
  v91 = (unsigned __int64)(v13 * v16 + 7) >> 3;
  v18 = 1;
  while ( 2 )
  {
    switch ( *(_BYTE *)(v17 + 8) )
    {
      case 1:
        v20 = 16;
        goto LABEL_18;
      case 2:
        v20 = 32;
        goto LABEL_18;
      case 3:
      case 9:
        v20 = 64;
        goto LABEL_18;
      case 4:
        v20 = 80;
        goto LABEL_18;
      case 5:
      case 6:
        v20 = 128;
        goto LABEL_18;
      case 7:
        v20 = 8 * (unsigned int)sub_15A9520((__int64)a5, 0);
        goto LABEL_18;
      case 0xB:
        v20 = *(_DWORD *)(v17 + 8) >> 8;
        goto LABEL_18;
      case 0xD:
        v20 = 8LL * *(_QWORD *)sub_15A9930((__int64)a5, v17);
        goto LABEL_18;
      case 0xE:
        v33 = *(_QWORD *)(v17 + 32);
        v81 = *(_QWORD *)(v17 + 24);
        v34 = sub_15A9FE0((__int64)a5, v81);
        v35 = v81;
        v36 = 1;
        v37 = v34;
        while ( 2 )
        {
          switch ( *(_BYTE *)(v35 + 8) )
          {
            case 1:
              v54 = 16;
              goto LABEL_78;
            case 2:
              v54 = 32;
              goto LABEL_78;
            case 3:
            case 9:
              v54 = 64;
              goto LABEL_78;
            case 4:
              v54 = 80;
              goto LABEL_78;
            case 5:
            case 6:
              v54 = 128;
              goto LABEL_78;
            case 7:
              v77 = v36;
              v61 = 0;
              v86 = v37;
              goto LABEL_92;
            case 0xB:
              v54 = *(_DWORD *)(v35 + 8) >> 8;
              goto LABEL_78;
            case 0xD:
              v79 = v36;
              v88 = v37;
              v64 = (_QWORD *)sub_15A9930((__int64)a5, v35);
              v37 = v88;
              v36 = v79;
              v54 = 8LL * *v64;
              goto LABEL_78;
            case 0xE:
              v71 = v36;
              v73 = v37;
              v75 = *(_QWORD *)(v35 + 24);
              v87 = *(_QWORD *)(v35 + 32);
              v78 = (unsigned int)sub_15A9FE0((__int64)a5, v75);
              v63 = sub_127FA20((__int64)a5, v75);
              v37 = v73;
              v36 = v71;
              v54 = 8 * v87 * v78 * ((v78 + ((unsigned __int64)(v63 + 7) >> 3) - 1) / v78);
              goto LABEL_78;
            case 0xF:
              v77 = v36;
              v86 = v37;
              v61 = *(_DWORD *)(v35 + 8) >> 8;
LABEL_92:
              v62 = sub_15A9520((__int64)a5, v61);
              v37 = v86;
              v36 = v77;
              v54 = (unsigned int)(8 * v62);
LABEL_78:
              v20 = 8 * v37 * v33 * ((v37 + ((unsigned __int64)(v54 * v36 + 7) >> 3) - 1) / v37);
              goto LABEL_18;
            case 0x10:
              v60 = *(_QWORD *)(v35 + 32);
              v35 = *(_QWORD *)(v35 + 24);
              v36 *= v60;
              continue;
            default:
              goto LABEL_106;
          }
        }
      case 0xF:
        v20 = 8 * (unsigned int)sub_15A9520((__int64)a5, *(_DWORD *)(v17 + 8) >> 8);
LABEL_18:
        v21 = *a1;
        v22 = *((_BYTE *)*a1 + 8);
        v23 = (unsigned __int64)(v20 * v18 + 7) >> 3;
        if ( v22 == 16 )
        {
          if ( *(_BYTE *)(*v21[2] + 8) != 15 )
            goto LABEL_51;
        }
        else if ( v22 != 15 )
        {
          goto LABEL_20;
        }
        v95 = 257;
        v40 = (__int64 **)sub_15A9650((__int64)a5, (__int64)v21);
        if ( v40 != *a1 )
        {
          if ( *((_BYTE *)a1 + 16) > 0x10u )
          {
            v97 = 257;
            v65 = sub_15FDBD0(45, (__int64)a1, (__int64)v40, (__int64)v96, 0);
            v5 = (__int64 ***)v65;
            if ( v99 )
            {
              v66 = v100;
              sub_157E9D0(v99 + 40, v65);
              v67 = v5[3];
              v68 = *v66;
              v5[4] = (__int64 **)v66;
              v68 &= 0xFFFFFFFFFFFFFFF8LL;
              v5[3] = (__int64 **)(v68 | (unsigned __int8)v67 & 7);
              *(_QWORD *)(v68 + 8) = v5 + 3;
              *v66 = *v66 & 7 | (unsigned __int64)(v5 + 3);
            }
            sub_164B780((__int64)v5, v94);
            if ( v98 )
            {
              v93 = v98;
              sub_1623A60((__int64)&v93, (__int64)v98, 2);
              v69 = (__int64)v5[6];
              if ( v69 )
                sub_161E7C0((__int64)(v5 + 6), v69);
              v70 = v93;
              v5[6] = (__int64 **)v93;
              if ( v70 )
                sub_1623210((__int64)&v93, v70, (__int64)(v5 + 6));
            }
          }
          else
          {
            v5 = (__int64 ***)sub_15A46C0(45, a1, v40, 0);
          }
        }
        v22 = *((_BYTE *)*v5 + 8);
LABEL_20:
        if ( v22 == 11 )
          goto LABEL_21;
LABEL_51:
        v95 = 257;
        v39 = (__int64 **)sub_1644900(v89, 8 * (int)v91);
        if ( v39 != *v5 )
        {
          if ( *((_BYTE *)v5 + 16) > 0x10u )
          {
            v97 = 257;
            v41 = sub_15FDBD0(47, (__int64)v5, (__int64)v39, (__int64)v96, 0);
            v5 = (__int64 ***)v41;
            if ( v99 )
            {
              v42 = v100;
              sub_157E9D0(v99 + 40, v41);
              v43 = v5[3];
              v44 = *v42;
              v5[4] = (__int64 **)v42;
              v44 &= 0xFFFFFFFFFFFFFFF8LL;
              v5[3] = (__int64 **)(v44 | (unsigned __int8)v43 & 7);
              *(_QWORD *)(v44 + 8) = v5 + 3;
              *v42 = *v42 & 7 | (unsigned __int64)(v5 + 3);
            }
            sub_164B780((__int64)v5, v94);
            if ( v98 )
            {
              v93 = v98;
              sub_1623A60((__int64)&v93, (__int64)v98, 2);
              v45 = (__int64)v5[6];
              if ( v45 )
                sub_161E7C0((__int64)(v5 + 6), v45);
              v46 = v93;
              v5[6] = (__int64 **)v93;
              if ( v46 )
                sub_1623210((__int64)&v93, v46, (__int64)(v5 + 6));
            }
          }
          else
          {
            v5 = (__int64 ***)sub_15A46C0(47, v5, v39, 0);
          }
        }
LABEL_21:
        if ( *a5 )
        {
          v24 = (unsigned int)(8 * (v91 - a2 - v23));
          if ( !(_DWORD)v24 )
            goto LABEL_23;
        }
        else
        {
          v24 = (unsigned int)(8 * a2);
          if ( !(_DWORD)v24 )
            goto LABEL_23;
        }
        v97 = 257;
        v38 = sub_15A0680((__int64)*v5, v24, 0);
        v5 = (__int64 ***)sub_156E320((__int64 *)&v98, (__int64)v5, v38, (__int64)v96, 0);
LABEL_23:
        if ( v91 != v23 )
        {
          v95 = 257;
          v25 = (__int64 **)sub_1644900(v89, 8 * (int)v23);
          if ( v25 != *v5 )
          {
            if ( *((_BYTE *)v5 + 16) > 0x10u )
            {
              v97 = 257;
              v47 = sub_15FDF30(v5, (__int64)v25, (__int64)v96, 0);
              v5 = (__int64 ***)v47;
              if ( v99 )
              {
                v48 = v100;
                sub_157E9D0(v99 + 40, v47);
                v49 = v5[3];
                v50 = *v48;
                v5[4] = (__int64 **)v48;
                v50 &= 0xFFFFFFFFFFFFFFF8LL;
                v5[3] = (__int64 **)(v50 | (unsigned __int8)v49 & 7);
                *(_QWORD *)(v50 + 8) = v5 + 3;
                *v48 = *v48 & 7 | (unsigned __int64)(v5 + 3);
              }
              sub_164B780((__int64)v5, v94);
              if ( v98 )
              {
                v93 = v98;
                sub_1623A60((__int64)&v93, (__int64)v98, 2);
                v51 = (__int64)v5[6];
                if ( v51 )
                  sub_161E7C0((__int64)(v5 + 6), v51);
                v52 = v93;
                v5[6] = (__int64 **)v93;
                if ( v52 )
                  sub_1623210((__int64)&v93, v52, (__int64)(v5 + 6));
              }
            }
            else
            {
              v5 = (__int64 ***)sub_15A4670(v5, v25);
            }
          }
        }
        break;
      case 0x10:
        v32 = *(_QWORD *)(v17 + 32);
        v17 = *(_QWORD *)(v17 + 24);
        v18 *= v32;
        continue;
      default:
LABEL_106:
        BUG();
    }
    break;
  }
LABEL_27:
  v26 = sub_1B710B0((__int64)v5, a3, (__int64 *)&v98, a5);
  if ( v98 )
    sub_161E7C0((__int64)&v98, (__int64)v98);
  return v26;
}
