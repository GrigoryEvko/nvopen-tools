// Function: sub_17AE700
// Address: 0x17ae700
//
__int64 __fastcall sub_17AE700(__int64 a1, __int64 ***a2, _DWORD *a3, __int64 a4)
{
  __int64 v5; // r9
  __int64 v6; // r14
  __int64 **v9; // rdx
  __int64 *v10; // rdi
  unsigned __int8 v11; // al
  unsigned int *v12; // r15
  __int64 v13; // r14
  __int64 v14; // rax
  int v15; // r8d
  __int64 v16; // r9
  __int64 v17; // rax
  unsigned int v18; // r13d
  _QWORD *v19; // rdi
  __int64 **v20; // rax
  int v21; // r8d
  int v22; // r9d
  __int64 v23; // r13
  __int64 v24; // rax
  __int64 *v25; // rdi
  __int64 v26; // rsi
  _BYTE *v27; // r14
  __int64 v28; // rax
  __int64 v29; // r14
  __int64 v31; // rdx
  char v32; // r14
  char v33; // dl
  __int64 v34; // r13
  unsigned __int64 v35; // r15
  __int64 ***v36; // rax
  __int64 v37; // rax
  __int64 v38; // rdx
  unsigned int v39; // ecx
  __int64 ***v40; // rsi
  bool v41; // al
  __int64 ****v42; // rsi
  __int64 ***v43; // rcx
  unsigned int v44; // r13d
  int v45; // edx
  __int64 v46; // r13
  __int64 *v47; // r15
  __int64 v48; // rax
  __int64 v49; // r13
  __int64 ***v50; // rdx
  __int64 v51; // rbx
  _QWORD *v52; // rax
  int v53; // eax
  _WORD *v54; // r14
  __int64 v55; // r8
  int v56; // edi
  __int64 v57; // rbx
  unsigned __int64 v58; // rax
  __int64 v59; // rdx
  bool v60; // al
  bool v61; // al
  bool v62; // al
  char v63; // al
  __int64 v64; // rsi
  __int16 v65; // ax
  __int64 *v66; // r15
  __int64 v67; // r14
  __int16 v68; // r13
  _QWORD **v69; // rax
  __int64 *v70; // rax
  __int64 v71; // rsi
  __int64 *v72; // rax
  __int64 *v73; // r14
  __int64 v74; // r15
  __int64 v75; // r13
  _QWORD *v76; // rax
  __int64 v77; // rbx
  __int64 v78; // rax
  __int64 *v79; // rax
  __int64 *v80; // rax
  int v81; // ecx
  __int64 *v82; // rdi
  __int64 *v83; // rsi
  __int64 *v84; // rax
  __int64 v85; // rdx
  __int64 *v86; // rax
  __int64 v87; // rdi
  bool v88; // al
  __int16 v89; // ax
  __int64 *v90; // r15
  __int64 v91; // r14
  __int16 v92; // r13
  _QWORD **v93; // rax
  __int64 *v94; // rax
  __int64 v95; // rsi
  __int64 **v96; // rax
  __int64 **v97; // rax
  __int64 v98; // rax
  __int64 *v99; // rax
  __int64 v100; // [rsp+8h] [rbp-108h]
  int v101; // [rsp+8h] [rbp-108h]
  __int64 ***v102; // [rsp+10h] [rbp-100h]
  unsigned int v103; // [rsp+10h] [rbp-100h]
  int v104; // [rsp+10h] [rbp-100h]
  __int64 v105; // [rsp+18h] [rbp-F8h]
  _DWORD *v106; // [rsp+18h] [rbp-F8h]
  __int64 v107; // [rsp+18h] [rbp-F8h]
  __int64 v108; // [rsp+18h] [rbp-F8h]
  __int64 v109; // [rsp+18h] [rbp-F8h]
  __int64 v110; // [rsp+20h] [rbp-F0h]
  _DWORD *v111; // [rsp+20h] [rbp-F0h]
  __int64 v112; // [rsp+20h] [rbp-F0h]
  _QWORD *v113; // [rsp+20h] [rbp-F0h]
  __int64 v114; // [rsp+20h] [rbp-F0h]
  _QWORD *v115; // [rsp+20h] [rbp-F0h]
  _BYTE v116[16]; // [rsp+30h] [rbp-E0h] BYREF
  __int16 v117; // [rsp+40h] [rbp-D0h]
  _WORD *v118; // [rsp+50h] [rbp-C0h] BYREF
  __int64 v119; // [rsp+58h] [rbp-B8h]
  _WORD v120[88]; // [rsp+60h] [rbp-B0h] BYREF

  v5 = a4;
  v6 = (unsigned int)(a4 - 1);
  while ( 2 )
  {
    v9 = *a2;
    v10 = (__int64 *)*a2;
    if ( *((_BYTE *)*a2 + 8) == 16 )
      v10 = (__int64 *)*v9[2];
    v11 = *((_BYTE *)a2 + 16);
    if ( v11 == 9 )
    {
      v96 = (__int64 **)sub_16463B0(v10, v5);
      return sub_1599EF0(v96);
    }
    if ( v11 == 10 )
    {
      v97 = (__int64 **)sub_16463B0(v10, v5);
      return sub_1598F00(v97);
    }
    if ( v11 > 0x10u )
    {
      switch ( *((_BYTE *)a2 + 16) )
      {
        case '#':
        case '$':
        case '%':
        case '&':
        case '\'':
        case '(':
        case ')':
        case '*':
        case '+':
        case ',':
        case '-':
        case '.':
        case '/':
        case '0':
        case '1':
        case '2':
        case '3':
        case '4':
        case '8':
        case '<':
        case '=':
        case '>':
        case '?':
        case '@':
        case 'A':
        case 'B':
        case 'C':
        case 'D':
        case 'K':
        case 'L':
        case 'O':
          v118 = v120;
          v119 = 0x800000000LL;
          v31 = *((unsigned int *)v9 + 8);
          v32 = v5 != v31;
          if ( (*((_DWORD *)a2 + 5) & 0xFFFFFFF) != 0 )
          {
            v33 = *((_BYTE *)a2 + 23);
            v34 = 24LL * (*((_DWORD *)a2 + 5) & 0xFFFFFFF);
            v35 = 0;
            do
            {
              if ( (v33 & 0x40) != 0 )
                v36 = (__int64 ***)*(a2 - 1);
              else
                v36 = &a2[-3 * (*((_DWORD *)a2 + 5) & 0xFFFFFFF)];
              v106 = a3;
              v110 = v5;
              v37 = sub_17AE700(a1, v36[v35 / 8], a3, v5);
              v38 = (unsigned int)v119;
              v5 = v110;
              a3 = v106;
              if ( (unsigned int)v119 >= HIDWORD(v119) )
              {
                v100 = v110;
                v112 = v37;
                sub_16CD150((__int64)&v118, v120, 0, 8, (int)v106, v5);
                v38 = (unsigned int)v119;
                v5 = v100;
                a3 = v106;
                v37 = v112;
              }
              *(_QWORD *)&v118[4 * v38] = v37;
              v39 = v119 + 1;
              LODWORD(v119) = v119 + 1;
              v33 = *((_BYTE *)a2 + 23);
              if ( (v33 & 0x40) != 0 )
                v40 = (__int64 ***)*(a2 - 1);
              else
                v40 = &a2[-3 * (*((_DWORD *)a2 + 5) & 0xFFFFFFF)];
              v41 = v40[v35 / 8] != (__int64 **)v37;
              v35 += 24LL;
              v32 |= v41;
            }
            while ( v34 != v35 );
            if ( !v32 )
              goto LABEL_31;
            v54 = v118;
            v55 = v39;
          }
          else
          {
            if ( v5 == v31 )
              return (__int64)a2;
            v54 = v120;
            v55 = 0;
            v39 = 0;
          }
          v56 = *((unsigned __int8 *)a2 + 16) - 24;
          switch ( *((_BYTE *)a2 + 16) )
          {
            case '#':
            case '$':
            case '%':
            case '&':
            case '\'':
            case '(':
            case ')':
            case '*':
            case '+':
            case ',':
            case '-':
            case '.':
            case '/':
            case '0':
            case '1':
            case '2':
            case '3':
            case '4':
              v117 = 257;
              v57 = sub_15FB440(v56, *(__int64 **)v54, *((_QWORD *)v54 + 1), (__int64)v116, (__int64)a2);
              v58 = *((unsigned __int8 *)a2 + 16);
              if ( (unsigned __int8)v58 <= 0x2Fu )
              {
                v59 = 0x80A800000000LL;
                if ( _bittest64(&v59, v58) )
                {
                  v60 = sub_15F2370((__int64)a2);
                  sub_15F2310(v57, v60);
                  v61 = sub_15F2380((__int64)a2);
                  sub_15F2330(v57, v61);
                  LODWORD(v58) = *((unsigned __int8 *)a2 + 16);
                }
              }
              if ( (unsigned __int8)(v58 - 48) <= 1u || (unsigned int)(v58 - 41) <= 1 )
              {
                v62 = sub_15F23D0((__int64)a2);
                sub_15F2350(v57, v62);
              }
              v63 = *((_BYTE *)*a2 + 8);
              if ( v63 == 16 )
                v63 = *(_BYTE *)(*(*a2)[2] + 8);
              if ( (unsigned __int8)(v63 - 1) > 5u && *((_BYTE *)a2 + 16) != 76 )
                goto LABEL_71;
              v64 = (__int64)a2;
              a2 = (__int64 ***)v57;
              sub_15F2500(v57, v64);
              break;
            case '5':
            case '6':
            case '7':
            case '9':
            case ':':
            case ';':
            case 'E':
            case 'F':
            case 'G':
            case 'H':
            case 'I':
            case 'J':
            case 'L':
              v65 = *((_WORD *)a2 + 9);
              v66 = *(__int64 **)v54;
              v67 = *((_QWORD *)v54 + 1);
              v117 = 257;
              HIBYTE(v65) &= ~0x80u;
              v68 = v65;
              v57 = (__int64)sub_1648A60(56, 2u);
              if ( v57 )
              {
                v69 = (_QWORD **)*v66;
                if ( *(_BYTE *)(*v66 + 8) == 16 )
                {
                  v113 = v69[4];
                  v70 = (__int64 *)sub_1643320(*v69);
                  v71 = (__int64)sub_16463B0(v70, (unsigned int)v113);
                }
                else
                {
                  v71 = sub_1643320(*v69);
                }
                sub_15FEC10(v57, v71, 52, v68, (__int64)v66, v67, (__int64)v116, (__int64)a2);
              }
              goto LABEL_71;
            case '8':
              v72 = *(__int64 **)v54;
              v73 = (__int64 *)(v54 + 4);
              v117 = 257;
              v74 = (__int64)a2[7];
              v114 = (__int64)v72;
              v75 = v55 - 1;
              if ( !v74 )
              {
                v98 = *v72;
                if ( *(_BYTE *)(*(_QWORD *)v114 + 8LL) == 16 )
                  v98 = **(_QWORD **)(v98 + 16);
                v74 = *(_QWORD *)(v98 + 24);
              }
              v108 = v55;
              v103 = v39;
              v76 = sub_1648A60(72, v39);
              v77 = (__int64)v76;
              if ( !v76 )
                goto LABEL_83;
              v109 = (__int64)&v76[-3 * v108];
              v78 = *(_QWORD *)v114;
              if ( *(_BYTE *)(*(_QWORD *)v114 + 8LL) == 16 )
                v78 = **(_QWORD **)(v78 + 16);
              v101 = v103;
              v104 = *(_DWORD *)(v78 + 8) >> 8;
              v79 = (__int64 *)sub_15F9F50(v74, (__int64)v73, v75);
              v80 = (__int64 *)sub_1646BA0(v79, v104);
              v81 = v101;
              v82 = v80;
              if ( *(_BYTE *)(*(_QWORD *)v114 + 8LL) == 16 )
              {
                v99 = sub_16463B0(v80, *(_QWORD *)(*(_QWORD *)v114 + 32LL));
                v81 = v101;
                v82 = v99;
              }
              else
              {
                v83 = &v73[v75];
                if ( v73 == v83 )
                  goto LABEL_82;
                v84 = v73;
                while ( 1 )
                {
                  v85 = *(_QWORD *)*v84;
                  if ( *(_BYTE *)(v85 + 8) == 16 )
                    break;
                  if ( v83 == ++v84 )
                    goto LABEL_82;
                }
                v86 = sub_16463B0(v82, *(_QWORD *)(v85 + 32));
                v81 = v101;
                v82 = v86;
              }
LABEL_82:
              sub_15F1EA0(v77, (__int64)v82, 32, v109, v81, (__int64)a2);
              *(_QWORD *)(v77 + 56) = v74;
              *(_QWORD *)(v77 + 64) = sub_15F9F50(v74, (__int64)v73, v75);
              sub_15F9CE0(v77, v114, v73, v75, (__int64)v116);
LABEL_83:
              v87 = (__int64)a2;
              a2 = (__int64 ***)v77;
              v88 = sub_15FA300(v87);
              sub_15FA2E0(v77, v88);
              break;
            case '<':
            case '=':
            case '>':
            case '?':
            case '@':
            case 'A':
            case 'B':
            case 'C':
            case 'D':
              JUMPOUT(0x17AED39);
            case 'K':
              v89 = *((_WORD *)a2 + 9);
              v90 = *(__int64 **)v54;
              v91 = *((_QWORD *)v54 + 1);
              v117 = 257;
              HIBYTE(v89) &= ~0x80u;
              v92 = v89;
              v57 = (__int64)sub_1648A60(56, 2u);
              if ( v57 )
              {
                v93 = (_QWORD **)*v90;
                if ( *(_BYTE *)(*v90 + 8) == 16 )
                {
                  v115 = v93[4];
                  v94 = (__int64 *)sub_1643320(*v93);
                  v95 = (__int64)sub_16463B0(v94, (unsigned int)v115);
                }
                else
                {
                  v95 = sub_1643320(*v93);
                }
                sub_15FEC10(v57, v95, 51, v92, (__int64)v90, v91, (__int64)v116, (__int64)a2);
              }
LABEL_71:
              a2 = (__int64 ***)v57;
              break;
          }
LABEL_31:
          if ( v118 != v120 )
            _libc_free((unsigned __int64)v118);
          return (__int64)a2;
        case '5':
        case '6':
        case '7':
        case '9':
        case ':':
        case ';':
        case 'E':
        case 'F':
        case 'G':
        case 'H':
        case 'I':
        case 'J':
        case 'M':
        case 'N':
        case 'P':
        case 'Q':
        case 'R':
        case 'S':
        case 'T':
          if ( (*((_BYTE *)a2 + 23) & 0x40) != 0 )
            v42 = (__int64 ****)*(a2 - 1);
          else
            v42 = (__int64 ****)&a2[-3 * (*((_DWORD *)a2 + 5) & 0xFFFFFFF)];
          v43 = v42[6];
          v44 = *((_DWORD *)v43 + 8);
          if ( v44 > 0x40 )
          {
            v107 = v5;
            v111 = a3;
            v102 = v42[6];
            v53 = sub_16A57B0((__int64)(v43 + 3));
            v45 = -1;
            a3 = v111;
            v5 = v107;
            if ( v44 - v53 <= 0x40 )
              v45 = *(_DWORD *)v102[3];
          }
          else
          {
            v45 = *((_DWORD *)v43 + 6);
          }
          if ( !(_DWORD)v5 )
            goto LABEL_49;
          v46 = 0;
          while ( 2 )
          {
            if ( a3[v46] == v45 )
            {
              v47 = (__int64 *)sub_17AE700(a1, *v42, a3, v5);
              v120[0] = 257;
              v48 = sub_1643350(*(_QWORD **)(*(_QWORD *)(a1 + 8) + 24LL));
              v49 = sub_159C470(v48, v46, 0);
              if ( (*((_BYTE *)a2 + 23) & 0x40) != 0 )
                v50 = (__int64 ***)*(a2 - 1);
              else
                v50 = &a2[-3 * (*((_DWORD *)a2 + 5) & 0xFFFFFFF)];
              v51 = (__int64)v50[3];
              v52 = sub_1648A60(56, 3u);
              v29 = (__int64)v52;
              if ( v52 )
                sub_15FA480((__int64)v52, v47, v51, v49, (__int64)&v118, (__int64)a2);
              return v29;
            }
            if ( v6 != v46 )
            {
              ++v46;
              continue;
            }
            break;
          }
LABEL_49:
          a2 = *v42;
          continue;
      }
    }
    break;
  }
  v118 = v120;
  v119 = 0x1000000000LL;
  if ( (_DWORD)v5 )
  {
    v12 = a3;
    v13 = (__int64)&a3[(unsigned int)(v5 - 1) + 1];
    do
    {
      while ( 1 )
      {
        v18 = *v12;
        v19 = *(_QWORD **)(*(_QWORD *)(a1 + 8) + 24LL);
        if ( *v12 != -1 )
          break;
        v20 = (__int64 **)sub_1643350(v19);
        v23 = sub_1599EF0(v20);
        v24 = (unsigned int)v119;
        if ( (unsigned int)v119 >= HIDWORD(v119) )
        {
          sub_16CD150((__int64)&v118, v120, 0, 8, v21, v22);
          v24 = (unsigned int)v119;
        }
        ++v12;
        *(_QWORD *)&v118[4 * v24] = v23;
        LODWORD(v119) = v119 + 1;
        if ( (unsigned int *)v13 == v12 )
          goto LABEL_16;
      }
      v14 = sub_1643350(v19);
      v16 = sub_159C470(v14, v18, 0);
      v17 = (unsigned int)v119;
      if ( (unsigned int)v119 >= HIDWORD(v119) )
      {
        v105 = v16;
        sub_16CD150((__int64)&v118, v120, 0, 8, v15, v16);
        v17 = (unsigned int)v119;
        v16 = v105;
      }
      ++v12;
      *(_QWORD *)&v118[4 * v17] = v16;
      LODWORD(v119) = v119 + 1;
    }
    while ( (unsigned int *)v13 != v12 );
LABEL_16:
    v25 = (__int64 *)v118;
    v26 = (unsigned int)v119;
  }
  else
  {
    v25 = (__int64 *)v120;
    v26 = 0;
  }
  v27 = (_BYTE *)sub_15A01B0(v25, v26);
  v28 = sub_1599EF0(*a2);
  v29 = sub_15A3950((__int64)a2, v28, v27, 0);
  if ( v118 != v120 )
    _libc_free((unsigned __int64)v118);
  return v29;
}
