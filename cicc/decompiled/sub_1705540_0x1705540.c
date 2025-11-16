// Function: sub_1705540
// Address: 0x1705540
//
__int64 __fastcall sub_1705540(__int64 a1, __int64 a2, signed __int64 a3, __int64 a4)
{
  __int64 v6; // r14
  unsigned __int64 v7; // rax
  __int64 v9; // rdx
  __int64 v10; // r13
  __int64 v11; // rax
  __int64 v12; // r15
  __int64 v13; // rsi
  unsigned int v14; // edx
  __int64 v15; // rax
  __int64 v16; // rsi
  signed __int64 v17; // rcx
  __int64 v18; // rax
  __int64 v19; // rdx
  int v20; // r8d
  int v21; // r9d
  __int64 v22; // r13
  __int64 v23; // rax
  __int64 v24; // r15
  __int64 v25; // rsi
  __int64 v26; // rcx
  __int64 v27; // rdi
  __int64 v28; // rax
  char v29; // al
  __int64 v30; // rbx
  __int64 v31; // rax
  __int64 v32; // rax
  int v33; // r8d
  __int64 v34; // r9
  __int64 v35; // r10
  __int64 v36; // rax
  __int64 v37; // rax
  int v38; // eax
  unsigned int v39; // eax
  __int64 v40; // rsi
  __int64 v41; // rdx
  unsigned __int64 v42; // r10
  _QWORD *v43; // rax
  int v44; // eax
  int v45; // eax
  __int64 v46; // rax
  int v47; // eax
  unsigned __int64 v48; // rax
  _QWORD *v49; // rax
  __int64 v50; // rbx
  __int64 v51; // rdi
  __int64 v52; // rsi
  unsigned __int64 v53; // rcx
  __int64 v54; // rax
  unsigned __int64 v55; // rcx
  unsigned __int64 v56; // rtt
  __int64 v57; // rbx
  int v58; // r8d
  int v59; // r9d
  __int64 v60; // rax
  __int64 v61; // rax
  int v62; // eax
  unsigned __int64 v63; // rax
  _QWORD *v64; // rax
  int v65; // eax
  __int64 v66; // rax
  unsigned int v67; // esi
  int v68; // eax
  unsigned __int64 v69; // rax
  _QWORD *v70; // rax
  __int64 v71; // rax
  __int64 v72; // [rsp+8h] [rbp-68h]
  __int64 v73; // [rsp+10h] [rbp-60h]
  unsigned __int64 v74; // [rsp+10h] [rbp-60h]
  __int64 v75; // [rsp+18h] [rbp-58h]
  unsigned __int64 v76; // [rsp+18h] [rbp-58h]
  __int64 v77; // [rsp+18h] [rbp-58h]
  unsigned __int64 v78; // [rsp+18h] [rbp-58h]
  unsigned int v79; // [rsp+20h] [rbp-50h]
  unsigned __int64 v80; // [rsp+20h] [rbp-50h]
  __int64 v81; // [rsp+20h] [rbp-50h]
  __int64 v82; // [rsp+20h] [rbp-50h]
  __int64 v83; // [rsp+20h] [rbp-50h]
  __int64 v84; // [rsp+20h] [rbp-50h]
  __int64 v85; // [rsp+28h] [rbp-48h]
  __int64 v86; // [rsp+30h] [rbp-40h]
  __int64 v87; // [rsp+30h] [rbp-40h]
  __int64 v88; // [rsp+30h] [rbp-40h]
  __int64 v89; // [rsp+30h] [rbp-40h]
  __int64 v90; // [rsp+30h] [rbp-40h]
  unsigned int v91; // [rsp+30h] [rbp-40h]
  unsigned int v92; // [rsp+30h] [rbp-40h]
  __int64 v93; // [rsp+30h] [rbp-40h]
  unsigned int v94; // [rsp+30h] [rbp-40h]
  unsigned __int64 v95; // [rsp+30h] [rbp-40h]
  __int64 v96; // [rsp+30h] [rbp-40h]
  unsigned __int64 v97; // [rsp+30h] [rbp-40h]
  unsigned __int64 v98; // [rsp+30h] [rbp-40h]
  __int64 v99; // [rsp+30h] [rbp-40h]

  v6 = *(_QWORD *)(a2 + 24);
  v7 = *(unsigned __int8 *)(v6 + 8);
  if ( (unsigned __int8)v7 > 0xFu || (v9 = 35454, !_bittest64(&v9, v7)) )
  {
    if ( (unsigned int)(v7 - 13) > 1 && (_DWORD)v7 != 16 || !sub_16435F0(v6, 0) )
      return 0;
  }
  v10 = 1;
  v11 = sub_15A9730(*(_QWORD *)(a1 + 2664), a2);
  v12 = *(_QWORD *)(a1 + 2664);
  v85 = v11;
  v13 = v6;
  v14 = sub_15A9FE0(v12, v6);
  while ( 2 )
  {
    switch ( *(_BYTE *)(v13 + 8) )
    {
      case 0:
      case 8:
      case 0xA:
      case 0xC:
      case 0x10:
        v46 = *(_QWORD *)(v13 + 32);
        v13 = *(_QWORD *)(v13 + 24);
        v10 *= v46;
        continue;
      case 1:
        v15 = 16;
        break;
      case 2:
        v15 = 32;
        break;
      case 3:
      case 9:
        v15 = 64;
        break;
      case 4:
        v15 = 80;
        break;
      case 5:
      case 6:
        v15 = 128;
        break;
      case 7:
        v91 = v14;
        v45 = sub_15A9520(v12, 0);
        v14 = v91;
        v15 = (unsigned int)(8 * v45);
        break;
      case 0xB:
        v15 = *(_DWORD *)(v13 + 8) >> 8;
        break;
      case 0xD:
        v94 = v14;
        v49 = (_QWORD *)sub_15A9930(v12, v13);
        v14 = v94;
        v15 = 8LL * *v49;
        break;
      case 0xE:
        v79 = v14;
        v93 = *(_QWORD *)(v13 + 32);
        v48 = sub_12BE0A0(v12, *(_QWORD *)(v13 + 24));
        v14 = v79;
        v15 = 8 * v93 * v48;
        break;
      case 0xF:
        v92 = v14;
        v47 = sub_15A9520(v12, *(_DWORD *)(v13 + 8) >> 8);
        v14 = v92;
        v15 = (unsigned int)(8 * v47);
        break;
    }
    break;
  }
  v16 = 0;
  v17 = (v14 + ((unsigned __int64)(v10 * v15 + 7) >> 3) - 1) / v14 * v14;
  if ( v17 )
  {
    v18 = a3 / v17;
    v19 = a3 % v17;
    a3 = v19;
    v16 = v18;
    if ( v19 < 0 )
    {
      a3 = v17 + v19;
      v16 = v18 - 1;
    }
  }
  v22 = sub_15A0680(v85, v16, 0);
  v23 = *(unsigned int *)(a4 + 8);
  if ( (unsigned int)v23 >= *(_DWORD *)(a4 + 12) )
  {
    sub_16CD150(a4, (const void *)(a4 + 16), 0, 8, v20, v21);
    v23 = *(unsigned int *)(a4 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a4 + 8 * v23) = v22;
  ++*(_DWORD *)(a4 + 8);
  if ( a3 )
  {
    v24 = a4;
    while ( 1 )
    {
      v25 = v6;
      v26 = 1;
      v27 = *(_QWORD *)(a1 + 2664);
      while ( 2 )
      {
        switch ( *(_BYTE *)(v25 + 8) )
        {
          case 1:
            v28 = 16;
            goto LABEL_22;
          case 2:
            v28 = 32;
            goto LABEL_22;
          case 3:
          case 9:
            v28 = 64;
            goto LABEL_22;
          case 4:
            v28 = 80;
            goto LABEL_22;
          case 5:
          case 6:
            v28 = 128;
            goto LABEL_22;
          case 7:
            v90 = v26;
            v44 = sub_15A9520(v27, 0);
            v26 = v90;
            v28 = (unsigned int)(8 * v44);
            goto LABEL_22;
          case 0xB:
            v28 = *(_DWORD *)(v25 + 8) >> 8;
            goto LABEL_22;
          case 0xD:
            v89 = v26;
            v43 = (_QWORD *)sub_15A9930(v27, v25);
            v26 = v89;
            v28 = 8LL * *v43;
            goto LABEL_22;
          case 0xE:
            v73 = v26;
            v75 = *(_QWORD *)(v25 + 24);
            v88 = *(_QWORD *)(v25 + 32);
            v39 = sub_15A9FE0(v27, v75);
            v40 = v75;
            v41 = 1;
            v26 = v73;
            v42 = v39;
            while ( 2 )
            {
              switch ( *(_BYTE *)(v40 + 8) )
              {
                case 1:
                  v66 = 16;
                  goto LABEL_68;
                case 2:
                  v66 = 32;
                  goto LABEL_68;
                case 3:
                case 9:
                  v66 = 64;
                  goto LABEL_68;
                case 4:
                  v66 = 80;
                  goto LABEL_68;
                case 5:
                case 6:
                  v66 = 128;
                  goto LABEL_68;
                case 7:
                  v67 = 0;
                  v76 = v42;
                  v82 = v41;
                  goto LABEL_72;
                case 0xB:
                  v66 = *(_DWORD *)(v40 + 8) >> 8;
                  goto LABEL_68;
                case 0xD:
                  v78 = v42;
                  v84 = v41;
                  v70 = (_QWORD *)sub_15A9930(v27, v40);
                  v41 = v84;
                  v42 = v78;
                  v26 = v73;
                  v66 = 8LL * *v70;
                  goto LABEL_68;
                case 0xE:
                  v72 = v73;
                  v74 = v42;
                  v77 = v41;
                  v83 = *(_QWORD *)(v40 + 32);
                  v69 = sub_12BE0A0(v27, *(_QWORD *)(v40 + 24));
                  v41 = v77;
                  v42 = v74;
                  v26 = v72;
                  v66 = 8 * v83 * v69;
                  goto LABEL_68;
                case 0xF:
                  v76 = v42;
                  v82 = v41;
                  v67 = *(_DWORD *)(v40 + 8) >> 8;
LABEL_72:
                  v68 = sub_15A9520(v27, v67);
                  v41 = v82;
                  v42 = v76;
                  v26 = v73;
                  v66 = (unsigned int)(8 * v68);
LABEL_68:
                  v28 = 8 * v88 * v42 * ((v42 + ((unsigned __int64)(v66 * v41 + 7) >> 3) - 1) / v42);
                  goto LABEL_22;
                case 0x10:
                  v71 = *(_QWORD *)(v40 + 32);
                  v40 = *(_QWORD *)(v40 + 24);
                  v41 *= v71;
                  continue;
                default:
                  goto LABEL_10;
              }
            }
          case 0xF:
            v87 = v26;
            v38 = sub_15A9520(v27, *(_DWORD *)(v25 + 8) >> 8);
            v26 = v87;
            v28 = (unsigned int)(8 * v38);
LABEL_22:
            if ( 8 * a3 >= (unsigned __int64)(v28 * v26) )
              return 0;
            v29 = *(_BYTE *)(v6 + 8);
            if ( v29 != 13 )
            {
              if ( v29 == 14 )
              {
                v50 = 1;
                v51 = *(_QWORD *)(a1 + 2664);
                v52 = *(_QWORD *)(v6 + 24);
                v53 = (unsigned int)sub_15A9FE0(v51, v52);
                while ( 2 )
                {
                  switch ( *(_BYTE *)(v52 + 8) )
                  {
                    case 1:
                      v54 = 16;
                      goto LABEL_53;
                    case 2:
                      v54 = 32;
                      goto LABEL_53;
                    case 3:
                    case 9:
                      v54 = 64;
                      goto LABEL_53;
                    case 4:
                      v54 = 80;
                      goto LABEL_53;
                    case 5:
                    case 6:
                      v54 = 128;
                      goto LABEL_53;
                    case 7:
                      v98 = v53;
                      v65 = sub_15A9520(v51, 0);
                      v53 = v98;
                      v54 = (unsigned int)(8 * v65);
                      goto LABEL_53;
                    case 0xB:
                      v54 = *(_DWORD *)(v52 + 8) >> 8;
                      goto LABEL_53;
                    case 0xD:
                      v97 = v53;
                      v64 = (_QWORD *)sub_15A9930(v51, v52);
                      v53 = v97;
                      v54 = 8LL * *v64;
                      goto LABEL_53;
                    case 0xE:
                      v80 = v53;
                      v96 = *(_QWORD *)(v52 + 32);
                      v63 = sub_12BE0A0(v51, *(_QWORD *)(v52 + 24));
                      v53 = v80;
                      v54 = 8 * v96 * v63;
                      goto LABEL_53;
                    case 0xF:
                      v95 = v53;
                      v62 = sub_15A9520(v51, *(_DWORD *)(v52 + 8) >> 8);
                      v53 = v95;
                      v54 = (unsigned int)(8 * v62);
LABEL_53:
                      v55 = (v53 + ((unsigned __int64)(v54 * v50 + 7) >> 3) - 1) / v53 * v53;
                      v56 = a3;
                      a3 %= v55;
                      v57 = sub_15A0680(v85, v56 / v55, 0);
                      v60 = *(unsigned int *)(v24 + 8);
                      if ( (unsigned int)v60 >= *(_DWORD *)(v24 + 12) )
                      {
                        sub_16CD150(v24, (const void *)(v24 + 16), 0, 8, v58, v59);
                        v60 = *(unsigned int *)(v24 + 8);
                      }
                      *(_QWORD *)(*(_QWORD *)v24 + 8 * v60) = v57;
                      ++*(_DWORD *)(v24 + 8);
                      v6 = *(_QWORD *)(v6 + 24);
                      goto LABEL_27;
                    case 0x10:
                      v61 = *(_QWORD *)(v52 + 32);
                      v52 = *(_QWORD *)(v52 + 24);
                      v50 *= v61;
                      continue;
                    default:
                      goto LABEL_10;
                  }
                }
              }
              return 0;
            }
            v86 = sub_15A9930(*(_QWORD *)(a1 + 2664), v6);
            v30 = (unsigned int)sub_15A8020(v86, a3);
            v31 = sub_1643350(*(_QWORD **)v6);
            v32 = sub_159C470(v31, v30, 0);
            v34 = v86;
            v35 = v32;
            v36 = *(unsigned int *)(v24 + 8);
            if ( (unsigned int)v36 >= *(_DWORD *)(v24 + 12) )
            {
              v81 = v86;
              v99 = v35;
              sub_16CD150(v24, (const void *)(v24 + 16), 0, 8, v33, v34);
              v36 = *(unsigned int *)(v24 + 8);
              v34 = v81;
              v35 = v99;
            }
            *(_QWORD *)(*(_QWORD *)v24 + 8 * v36) = v35;
            ++*(_DWORD *)(v24 + 8);
            a3 -= *(_QWORD *)(v34 + 8 * v30 + 16);
            v6 = *(_QWORD *)(*(_QWORD *)(v6 + 16) + 8 * v30);
LABEL_27:
            if ( !a3 )
              return v6;
            break;
          case 0x10:
            v37 = *(_QWORD *)(v25 + 32);
            v25 = *(_QWORD *)(v25 + 24);
            v26 *= v37;
            continue;
          default:
LABEL_10:
            ++*(_DWORD *)(a3 + 40);
            BUG();
        }
        break;
      }
    }
  }
  return v6;
}
