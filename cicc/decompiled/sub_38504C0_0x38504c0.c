// Function: sub_38504C0
// Address: 0x38504c0
//
__int64 __fastcall sub_38504C0(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // rbx
  char v3; // al
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // r12
  int v7; // r12d
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r12
  int v14; // r12d
  __int64 v15; // rax
  __int64 v16; // rdx
  unsigned __int64 v17; // rax
  int v18; // r14d
  __int64 v19; // r13
  __int64 v20; // rax
  _QWORD *v21; // rdi
  __int64 *v22; // rax
  __int64 v23; // r8
  __int64 v24; // rsi
  __int64 v25; // r12
  int v26; // eax
  unsigned int v27; // eax
  __int64 v29; // rax
  int v30; // eax
  int v31; // eax
  __int64 v32; // r12
  unsigned int v33; // eax
  __int64 v34; // rsi
  __int64 v35; // r9
  unsigned __int64 v36; // r10
  _QWORD *v37; // rax
  __int64 v38; // rax
  __int64 v39; // rax
  unsigned int v40; // eax
  __int64 v41; // rsi
  unsigned int v42; // esi
  int v43; // eax
  _QWORD *v44; // rax
  __int64 v45; // rax
  unsigned int v46; // esi
  int v47; // eax
  __int64 v48; // rax
  _QWORD *v49; // rax
  __int64 v50; // rax
  __int64 v51; // [rsp+8h] [rbp-A8h]
  __int64 v52; // [rsp+10h] [rbp-A0h]
  __int64 v53; // [rsp+18h] [rbp-98h]
  __int64 v54; // [rsp+18h] [rbp-98h]
  __int64 v55; // [rsp+20h] [rbp-90h]
  __int64 v56; // [rsp+20h] [rbp-90h]
  __int64 v57; // [rsp+20h] [rbp-90h]
  __int64 v58; // [rsp+28h] [rbp-88h]
  unsigned __int64 v59; // [rsp+28h] [rbp-88h]
  unsigned __int64 v60; // [rsp+28h] [rbp-88h]
  __int64 v61; // [rsp+30h] [rbp-80h]
  __int64 v62; // [rsp+30h] [rbp-80h]
  __int64 v63; // [rsp+30h] [rbp-80h]
  unsigned __int64 v64; // [rsp+38h] [rbp-78h]
  __int64 v65; // [rsp+38h] [rbp-78h]
  __int64 v66; // [rsp+38h] [rbp-78h]
  __int64 v67; // [rsp+40h] [rbp-70h]
  __int64 v68; // [rsp+40h] [rbp-70h]
  unsigned __int64 v69; // [rsp+40h] [rbp-70h]
  unsigned __int64 v70; // [rsp+40h] [rbp-70h]
  __int64 v71; // [rsp+48h] [rbp-68h]
  __int64 v72; // [rsp+48h] [rbp-68h]
  __int64 v73; // [rsp+48h] [rbp-68h]
  __int64 v74; // [rsp+48h] [rbp-68h]
  __int64 v75; // [rsp+48h] [rbp-68h]
  __int64 v76; // [rsp+48h] [rbp-68h]
  __int64 v77; // [rsp+48h] [rbp-68h]
  __int64 v79; // [rsp+58h] [rbp-58h]
  __int64 v80; // [rsp+60h] [rbp-50h]
  char v81; // [rsp+6Fh] [rbp-41h]
  _QWORD v82[7]; // [rsp+78h] [rbp-38h] BYREF

  v2 = a1 & 0xFFFFFFFFFFFFFFF8LL;
  v3 = *(_BYTE *)((a1 & 0xFFFFFFFFFFFFFFF8LL) + 23);
  v81 = (a1 >> 2) & 1;
  if ( v81 )
  {
    if ( v3 < 0 )
    {
      v4 = sub_1648A40(v2);
      v6 = v4 + v5;
      if ( *(char *)(v2 + 23) >= 0 )
      {
        if ( (unsigned int)(v6 >> 4) )
          goto LABEL_77;
      }
      else if ( (unsigned int)((v6 - sub_1648A40(v2)) >> 4) )
      {
        if ( *(char *)(v2 + 23) < 0 )
        {
          v7 = *(_DWORD *)(sub_1648A40(v2) + 8);
          if ( *(char *)(v2 + 23) >= 0 )
            BUG();
          v8 = sub_1648A40(v2);
          v10 = -24 - 24LL * (unsigned int)(*(_DWORD *)(v8 + v9 - 4) - v7);
          goto LABEL_16;
        }
LABEL_77:
        BUG();
      }
    }
    v10 = -24;
    goto LABEL_16;
  }
  if ( v3 >= 0 )
    goto LABEL_15;
  v11 = sub_1648A40(v2);
  v13 = v11 + v12;
  if ( *(char *)(v2 + 23) >= 0 )
  {
    if ( (unsigned int)(v13 >> 4) )
LABEL_78:
      BUG();
LABEL_15:
    v10 = -72;
    goto LABEL_16;
  }
  if ( !(unsigned int)((v13 - sub_1648A40(v2)) >> 4) )
    goto LABEL_15;
  if ( *(char *)(v2 + 23) >= 0 )
    goto LABEL_78;
  v14 = *(_DWORD *)(sub_1648A40(v2) + 8);
  if ( *(char *)(v2 + 23) >= 0 )
    BUG();
  v15 = sub_1648A40(v2);
  v10 = -72 - 24LL * (unsigned int)(*(_DWORD *)(v15 + v16 - 4) - v14);
LABEL_16:
  v17 = 0xAAAAAAAAAAAAAAABLL * ((v10 + 24LL * (*(_DWORD *)(v2 + 20) & 0xFFFFFFF)) >> 3);
  if ( (_DWORD)v17 )
  {
    v18 = 0;
    v80 = (unsigned int)v17;
    v19 = 0;
    while ( 1 )
    {
      v21 = (_QWORD *)(v2 + 56);
      if ( v81 )
      {
        if ( (unsigned __int8)sub_1560290(v21, v19, 6) )
          goto LABEL_28;
        v20 = *(_QWORD *)(v2 - 24);
        if ( !*(_BYTE *)(v20 + 16) )
          goto LABEL_20;
LABEL_21:
        v18 += 5;
        if ( v80 == ++v19 )
          return (unsigned int)(v18 + 30);
      }
      else
      {
        if ( (unsigned __int8)sub_1560290(v21, v19, 6) )
          goto LABEL_28;
        v20 = *(_QWORD *)(v2 - 72);
        if ( *(_BYTE *)(v20 + 16) )
          goto LABEL_21;
LABEL_20:
        v82[0] = *(_QWORD *)(v20 + 112);
        if ( !(unsigned __int8)sub_1560290(v82, v19, 6) )
          goto LABEL_21;
LABEL_28:
        v79 = 1;
        v22 = *(__int64 **)(v2 - 24LL * (*(_DWORD *)(v2 + 20) & 0xFFFFFFF) + 24 * v19);
        v23 = *v22;
        v24 = *(_QWORD *)(*v22 + 24);
        while ( 2 )
        {
          switch ( *(_BYTE *)(v24 + 8) )
          {
            case 0:
            case 8:
            case 0xA:
            case 0xC:
            case 0x10:
              v29 = v79 * *(_QWORD *)(v24 + 32);
              v24 = *(_QWORD *)(v24 + 24);
              v79 = v29;
              continue;
            case 1:
              LODWORD(v25) = 16;
              break;
            case 2:
              LODWORD(v25) = 32;
              break;
            case 3:
            case 9:
              LODWORD(v25) = 64;
              break;
            case 4:
              LODWORD(v25) = 80;
              break;
            case 5:
            case 6:
              LODWORD(v25) = 128;
              break;
            case 7:
              v71 = v23;
              v30 = sub_15A9520(a2, 0);
              v23 = v71;
              LODWORD(v25) = 8 * v30;
              break;
            case 0xB:
              LODWORD(v25) = *(_DWORD *)(v24 + 8) >> 8;
              break;
            case 0xD:
              v74 = v23;
              v37 = (_QWORD *)sub_15A9930(a2, v24);
              v23 = v74;
              v25 = 8LL * *v37;
              break;
            case 0xE:
              v32 = *(_QWORD *)(v24 + 32);
              v67 = v23;
              v73 = *(_QWORD *)(v24 + 24);
              v33 = sub_15A9FE0(a2, v73);
              v23 = v67;
              v34 = v73;
              v35 = 1;
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
                    v39 = *(_QWORD *)(v34 + 32);
                    v34 = *(_QWORD *)(v34 + 24);
                    v35 *= v39;
                    continue;
                  case 1:
                    v38 = 16;
                    goto LABEL_48;
                  case 2:
                    v38 = 32;
                    goto LABEL_48;
                  case 3:
                  case 9:
                    v38 = 64;
                    goto LABEL_48;
                  case 4:
                    v38 = 80;
                    goto LABEL_48;
                  case 5:
                  case 6:
                    v38 = 128;
                    goto LABEL_48;
                  case 7:
                    v65 = v67;
                    v42 = 0;
                    v69 = v36;
                    v76 = v35;
                    goto LABEL_56;
                  case 0xB:
                    v38 = *(_DWORD *)(v34 + 8) >> 8;
                    goto LABEL_48;
                  case 0xD:
                    v66 = v67;
                    v70 = v36;
                    v77 = v35;
                    v44 = (_QWORD *)sub_15A9930(a2, v34);
                    v35 = v77;
                    v36 = v70;
                    v23 = v66;
                    v38 = 8LL * *v44;
                    goto LABEL_48;
                  case 0xE:
                    v53 = v67;
                    v55 = v36;
                    v58 = v35;
                    v61 = *(_QWORD *)(v34 + 24);
                    v68 = *(_QWORD *)(v34 + 32);
                    v40 = sub_15A9FE0(a2, v61);
                    v23 = v53;
                    v36 = v55;
                    v75 = 1;
                    v41 = v61;
                    v35 = v58;
                    v64 = v40;
                    while ( 2 )
                    {
                      switch ( *(_BYTE *)(v41 + 8) )
                      {
                        case 0:
                        case 8:
                        case 0xA:
                        case 0xC:
                        case 0x10:
                          v50 = v75 * *(_QWORD *)(v41 + 32);
                          v41 = *(_QWORD *)(v41 + 24);
                          v75 = v50;
                          continue;
                        case 1:
                          v45 = 16;
                          goto LABEL_62;
                        case 2:
                          v45 = 32;
                          goto LABEL_62;
                        case 3:
                        case 9:
                          v45 = 64;
                          goto LABEL_62;
                        case 4:
                          v45 = 80;
                          goto LABEL_62;
                        case 5:
                        case 6:
                          v45 = 128;
                          goto LABEL_62;
                        case 7:
                          v56 = v53;
                          v46 = 0;
                          v59 = v36;
                          v62 = v35;
                          goto LABEL_66;
                        case 0xB:
                          v45 = *(_DWORD *)(v41 + 8) >> 8;
                          goto LABEL_62;
                        case 0xD:
                          v49 = (_QWORD *)sub_15A9930(a2, v41);
                          v35 = v58;
                          v36 = v55;
                          v23 = v53;
                          v45 = 8LL * *v49;
                          goto LABEL_62;
                        case 0xE:
                          v51 = v53;
                          v52 = v55;
                          v54 = v58;
                          v57 = *(_QWORD *)(v41 + 24);
                          v63 = *(_QWORD *)(v41 + 32);
                          v60 = (unsigned int)sub_15A9FE0(a2, v57);
                          v48 = sub_127FA20(a2, v57);
                          v35 = v54;
                          v36 = v52;
                          v23 = v51;
                          v45 = 8 * v63 * v60 * ((v60 + ((unsigned __int64)(v48 + 7) >> 3) - 1) / v60);
                          goto LABEL_62;
                        case 0xF:
                          v56 = v53;
                          v59 = v36;
                          v62 = v35;
                          v46 = *(_DWORD *)(v41 + 8) >> 8;
LABEL_66:
                          v47 = sub_15A9520(a2, v46);
                          v35 = v62;
                          v36 = v59;
                          v23 = v56;
                          v45 = (unsigned int)(8 * v47);
LABEL_62:
                          v38 = 8 * v64 * v68 * ((v64 + ((unsigned __int64)(v75 * v45 + 7) >> 3) - 1) / v64);
                          break;
                      }
                      goto LABEL_48;
                    }
                  case 0xF:
                    v65 = v67;
                    v69 = v36;
                    v76 = v35;
                    v42 = *(_DWORD *)(v34 + 8) >> 8;
LABEL_56:
                    v43 = sub_15A9520(a2, v42);
                    v35 = v76;
                    v36 = v69;
                    v23 = v65;
                    v38 = (unsigned int)(8 * v43);
LABEL_48:
                    v25 = 8 * v36 * v32 * ((v36 + ((unsigned __int64)(v38 * v35 + 7) >> 3) - 1) / v36);
                    break;
                }
                break;
              }
              break;
            case 0xF:
              v72 = v23;
              v31 = sub_15A9520(a2, *(_DWORD *)(v24 + 8) >> 8);
              v23 = v72;
              LODWORD(v25) = 8 * v31;
              break;
          }
          break;
        }
        v26 = sub_15A9520(a2, *(_DWORD *)(v23 + 8) >> 8);
        v27 = (8 * v26 + (int)v79 * (int)v25 - 1) / (unsigned int)(8 * v26);
        if ( v27 > 8 )
          v27 = 8;
        ++v19;
        v18 += 10 * v27;
        if ( v80 == v19 )
          return (unsigned int)(v18 + 30);
      }
    }
  }
  return 30;
}
