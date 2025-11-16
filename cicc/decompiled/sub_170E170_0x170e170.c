// Function: sub_170E170
// Address: 0x170e170
//
__int64 __fastcall sub_170E170(
        __int64 *a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  bool v10; // zf
  _QWORD *v11; // rax
  _QWORD *v12; // rdi
  unsigned int v13; // eax
  __int64 v14; // rdx
  __int64 v15; // rbx
  __int64 v16; // rbx
  int v17; // r8d
  int v18; // r9d
  _QWORD *v19; // r15
  __int64 v20; // rax
  __int64 v21; // r12
  unsigned __int64 *v22; // r13
  _QWORD *v23; // rbx
  _QWORD *v24; // r13
  __int64 v25; // rax
  int v27; // eax
  _QWORD *v28; // rdi
  __int64 v29; // rax
  __int64 *v30; // r15
  __int64 v31; // r14
  __int64 v32; // r13
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rbx
  __int64 v36; // r12
  _QWORD *v37; // rax
  double v38; // xmm4_8
  double v39; // xmm5_8
  _BYTE *v40; // r12
  __int64 v41; // rax
  signed __int64 *v42; // rax
  __int64 v43; // rbx
  __int64 v44; // r12
  char v45; // al
  unsigned __int8 v46; // r13
  _QWORD *v47; // rax
  __int64 v48; // rax
  __int64 v49; // rax
  __int64 v50; // r14
  __int64 v51; // r13
  _QWORD *v52; // rax
  double v53; // xmm4_8
  double v54; // xmm5_8
  __int64 *v55; // rax
  __int64 v56; // rax
  __int64 v57; // r9
  __int64 v58; // r14
  __int64 v59; // rbx
  __int64 v60; // r12
  _QWORD *v61; // rax
  __int64 v62; // r13
  __int64 *v63; // r12
  signed __int64 *v64; // rbx
  __int64 v65; // rsi
  int v66; // eax
  __int64 v67; // rdx
  __int64 v68; // rcx
  char v69; // al
  char v70; // al
  int v71; // eax
  _QWORD *v72; // rdi
  unsigned __int64 v73; // r12
  __int64 *v74; // r12
  __int64 v75; // rax
  __int64 v76; // rax
  __int64 v77; // rdx
  __int64 v78; // rax
  int v79; // eax
  bool v80; // al
  __int64 v81; // rax
  signed __int64 *v82; // r14
  __int64 v83; // r13
  signed __int64 *v84; // rbx
  __int64 v85; // rdi
  __int64 v86; // rax
  __int64 v87; // r14
  __int64 v88; // r13
  _QWORD *v89; // rax
  double v90; // xmm4_8
  double v91; // xmm5_8
  double v92; // xmm4_8
  double v93; // xmm5_8
  __int64 v94; // [rsp+0h] [rbp-6D0h]
  __int64 v95; // [rsp+0h] [rbp-6D0h]
  signed __int64 *v97; // [rsp+8h] [rbp-6C8h]
  unsigned __int64 v98; // [rsp+18h] [rbp-6B8h]
  __int64 v99; // [rsp+20h] [rbp-6B0h]
  signed __int64 v100; // [rsp+28h] [rbp-6A8h]
  int v101; // [rsp+30h] [rbp-6A0h]
  char v102; // [rsp+37h] [rbp-699h]
  __int64 v103; // [rsp+38h] [rbp-698h]
  _QWORD *v104; // [rsp+40h] [rbp-690h]
  __int64 ***v105; // [rsp+40h] [rbp-690h]
  __int64 v106; // [rsp+40h] [rbp-690h]
  __int64 v107; // [rsp+48h] [rbp-688h]
  __int64 v108; // [rsp+48h] [rbp-688h]
  __int64 ***v109; // [rsp+48h] [rbp-688h]
  __int64 v110; // [rsp+48h] [rbp-688h]
  __int64 v111; // [rsp+48h] [rbp-688h]
  __int64 v112; // [rsp+48h] [rbp-688h]
  __int64 v113; // [rsp+48h] [rbp-688h]
  signed __int64 v114; // [rsp+58h] [rbp-678h] BYREF
  signed __int64 v115; // [rsp+60h] [rbp-670h] BYREF
  __int64 v116; // [rsp+68h] [rbp-668h]
  _QWORD v117[4]; // [rsp+70h] [rbp-660h] BYREF
  _BYTE *v118; // [rsp+90h] [rbp-640h] BYREF
  __int64 v119; // [rsp+98h] [rbp-638h]
  _BYTE v120[1584]; // [rsp+A0h] [rbp-630h] BYREF

  v10 = *(_BYTE *)(a2 + 16) == 53;
  v118 = v120;
  v103 = a2;
  v119 = 0x4000000000LL;
  v114 = 0;
  if ( v10 )
  {
    sub_1AEA030(&v115);
    v73 = v115 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (v115 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
    {
      if ( (v115 & 4) != 0 && !*(_DWORD *)(v73 + 8) )
      {
        if ( *(_QWORD *)v73 != v73 + 16 )
          _libc_free(*(_QWORD *)v73);
        a2 = 48;
        j_j___libc_free_0(v73, 48);
        v102 = 0;
        v100 = 0;
      }
      else
      {
        v114 = v115;
        v100 = v115;
        v102 = (v115 >> 2) & 1;
      }
    }
    else
    {
      v102 = 0;
      v100 = 0;
    }
    v74 = (__int64 *)sub_15F2050(v103);
    v75 = sub_22077B0(464);
    v99 = v75;
    if ( v75 )
    {
      a2 = (__int64)v74;
      sub_15A5590(v75, v74, 0, 0);
    }
  }
  else
  {
    v102 = 0;
    v100 = 0;
    v99 = 0;
  }
  v11 = (_QWORD *)a1[331];
  v115 = (signed __int64)v117;
  v12 = v117;
  v104 = v11;
  v117[0] = v103;
  v116 = 0x400000001LL;
  v13 = 1;
LABEL_4:
  v14 = v13--;
  v15 = v12[v14 - 1];
  LODWORD(v116) = v13;
  if ( !*(_QWORD *)(v15 + 8) )
    goto LABEL_36;
  v107 = v15;
  v16 = *(_QWORD *)(v15 + 8);
  while ( 2 )
  {
    v19 = sub_1648700(v16);
    switch ( *((_BYTE *)v19 + 16) )
    {
      case '7':
        if ( (*((_BYTE *)v19 + 18) & 1) != 0 )
          goto LABEL_9;
        v20 = *(v19 - 3);
        if ( v107 != v20 || !v20 )
          goto LABEL_9;
        goto LABEL_95;
      case '8':
      case 'G':
      case 'H':
        v27 = v119;
        if ( (unsigned int)v119 >= HIDWORD(v119) )
        {
          a2 = 0;
          sub_170B450((__int64)&v118, 0);
          v27 = v119;
        }
        v28 = &v118[24 * v27];
        if ( v28 )
        {
          *v28 = 6;
          v28[1] = 0;
          v28[2] = v19;
          if ( v19 != (_QWORD *)-8LL && v19 != (_QWORD *)-16LL )
            sub_164C220((__int64)v28);
          v27 = v119;
        }
        LODWORD(v119) = v27 + 1;
        v29 = (unsigned int)v116;
        if ( (unsigned int)v116 >= HIDWORD(v116) )
        {
          a2 = (__int64)v117;
          sub_16CD150((__int64)&v115, v117, 0, 8, v17, v18);
          v29 = (unsigned int)v116;
        }
        *(_QWORD *)(v115 + 8 * v29) = v19;
        LODWORD(v116) = v116 + 1;
        goto LABEL_34;
      case 'K':
        v66 = *((unsigned __int16 *)v19 + 9);
        BYTE1(v66) &= ~0x80u;
        if ( (unsigned int)(v66 - 32) > 1 )
          goto LABEL_9;
        v67 = *(v19 - 6);
        v68 = v19[3 * ((v107 == v67) & (unsigned __int8)(v67 != 0)) - 6];
        v69 = *(_BYTE *)(v68 + 16);
        if ( v69 == 15 )
          goto LABEL_84;
        if ( v69 == 54 )
        {
          if ( *(_BYTE *)(*(_QWORD *)(v68 - 24) + 16LL) != 3 )
            goto LABEL_9;
        }
        else
        {
          a2 = (__int64)v104;
          v94 = v19[3 * ((v107 == v67) & (unsigned __int8)(v67 != 0)) - 6];
          v70 = sub_140B1C0(v94, v104, 0);
          if ( v103 == v94 || !v70 )
          {
LABEL_9:
            if ( (_QWORD *)v115 != v117 )
              _libc_free(v115);
            v21 = 0;
            goto LABEL_12;
          }
        }
LABEL_84:
        v71 = v119;
        if ( (unsigned int)v119 >= HIDWORD(v119) )
        {
          a2 = 0;
          sub_170B450((__int64)&v118, 0);
          v71 = v119;
        }
        v72 = &v118[24 * v71];
        if ( v72 )
        {
          *v72 = 6;
          v72[1] = 0;
          v72[2] = v19;
          if ( v19 != (_QWORD *)-8LL && v19 != (_QWORD *)-16LL )
LABEL_100:
            sub_164C220((__int64)v72);
LABEL_101:
          LODWORD(v119) = v119 + 1;
        }
        else
        {
LABEL_87:
          LODWORD(v119) = v71 + 1;
        }
LABEL_34:
        v16 = *(_QWORD *)(v16 + 8);
        if ( v16 )
          continue;
        v13 = v116;
        v12 = (_QWORD *)v115;
LABEL_36:
        if ( v13 )
          goto LABEL_4;
        v30 = a1;
        if ( v12 != v117 )
          _libc_free((unsigned __int64)v12);
        v98 = v100 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (_DWORD)v119 )
        {
          v31 = 0;
          v108 = 24LL * (unsigned int)v119;
          do
          {
            while ( 1 )
            {
              v32 = *(_QWORD *)&v118[v31 + 16];
              if ( v32 )
              {
                if ( *(_BYTE *)(v32 + 16) == 78 )
                {
                  v33 = *(_QWORD *)(v32 - 24);
                  if ( !*(_BYTE *)(v33 + 16) && (*(_BYTE *)(v33 + 33) & 0x20) != 0 && *(_DWORD *)(v33 + 36) == 144 )
                  {
                    v34 = sub_140EAC0(*(__int64 **)&v118[v31 + 16], a1[333], a1[331], 1);
                    v35 = *(_QWORD *)(v32 + 8);
                    v105 = (__int64 ***)v34;
                    if ( v35 )
                    {
                      v36 = *a1;
                      do
                      {
                        v37 = sub_1648700(v35);
                        sub_170B990(v36, (__int64)v37);
                        v35 = *(_QWORD *)(v35 + 8);
                      }
                      while ( v35 );
                      if ( v105 == (__int64 ***)v32 )
                        v105 = (__int64 ***)sub_1599EF0(*v105);
                      sub_164D160(v32, (__int64)v105, a3, a4, a5, a6, v38, v39, a9, a10);
                    }
                    sub_170BC50((__int64)a1, v32);
                    v40 = &v118[v31];
                    v41 = *(_QWORD *)&v118[v31 + 16];
                    if ( v41 )
                      break;
                  }
                }
              }
              v31 += 24;
              if ( v108 == v31 )
                goto LABEL_58;
            }
            if ( v41 != -16 && v41 != -8 )
              sub_1649B30(&v118[v31]);
            *((_QWORD *)v40 + 2) = 0;
            v31 += 24;
          }
          while ( v108 != v31 );
LABEL_58:
          if ( (_DWORD)v119 )
          {
            v106 = 24LL * (unsigned int)v119;
            v42 = &v115;
            if ( !v98 )
              v42 = &v114;
            v43 = 0;
            v97 = v42;
            while ( 1 )
            {
              v44 = *(_QWORD *)&v118[v43 + 16];
              if ( v44 )
                break;
LABEL_70:
              v43 += 24;
              if ( v106 == v43 )
                goto LABEL_71;
            }
            v45 = *(_BYTE *)(v44 + 16);
            if ( v45 == 75 )
            {
              v46 = sub_15FF850(*(_WORD *)(v44 + 18) & 0x7FFF);
              v47 = (_QWORD *)sub_16498A0(v44);
              v48 = sub_1643320(v47);
              v49 = sub_159C470(v48, v46, 0);
              v50 = *(_QWORD *)(v44 + 8);
              v109 = (__int64 ***)v49;
              if ( v50 )
              {
                v51 = *v30;
                do
                {
                  v52 = sub_1648700(v50);
                  sub_170B990(v51, (__int64)v52);
                  v50 = *(_QWORD *)(v50 + 8);
                }
                while ( v50 );
                if ( (__int64 ***)v44 != v109 )
                  goto LABEL_68;
                v113 = sub_1599EF0(*(__int64 ***)v44);
                sub_164D160(v44, v113, a3, a4, a5, a6, v92, v93, a9, a10);
              }
            }
            else if ( (unsigned __int8)(v45 - 71) <= 1u || v45 == 56 )
            {
              v86 = sub_1599EF0(*(__int64 ***)v44);
              v87 = *(_QWORD *)(v44 + 8);
              v109 = (__int64 ***)v86;
              if ( v87 )
              {
                v88 = *v30;
                do
                {
                  v89 = sub_1648700(v87);
                  sub_170B990(v88, (__int64)v89);
                  v87 = *(_QWORD *)(v87 + 8);
                }
                while ( v87 );
                if ( v109 == (__int64 ***)v44 )
                {
                  v112 = sub_1599EF0(*v109);
                  sub_164D160(v44, v112, a3, a4, a5, a6, v90, v91, a9, a10);
                }
                else
                {
LABEL_68:
                  sub_164D160(v44, (__int64)v109, a3, a4, a5, a6, v53, v54, a9, a10);
                }
              }
            }
            else if ( v45 == 55 )
            {
              if ( v102 )
              {
                v82 = *(signed __int64 **)v98;
                v83 = *(_QWORD *)v98 + 8LL * *(unsigned int *)(v98 + 8);
              }
              else
              {
                v83 = (__int64)v97;
                v82 = &v114;
              }
              if ( (signed __int64 *)v83 != v82 )
              {
                v111 = v43;
                v84 = v82;
                do
                {
                  v85 = *v84++;
                  sub_1AE9B50(v85, v44, v99);
                }
                while ( (signed __int64 *)v83 != v84 );
                v43 = v111;
              }
            }
            sub_170BC50((__int64)v30, v44);
            goto LABEL_70;
          }
        }
LABEL_71:
        if ( *(_BYTE *)(v103 + 16) == 29 )
        {
          v55 = (__int64 *)sub_15F2050(v103);
          v56 = sub_15E26F0(v55, 40, 0, 0);
          v57 = *(_QWORD *)(v103 + 40);
          v58 = v56;
          LOWORD(v117[0]) = 257;
          v59 = *(_QWORD *)(v103 - 24);
          v110 = v57;
          v60 = *(_QWORD *)(v103 - 48);
          v61 = sub_1648A60(72, 3u);
          v62 = (__int64)v61;
          if ( v61 )
          {
            sub_15F1F50(
              (__int64)v61,
              **(_QWORD **)(*(_QWORD *)(*(_QWORD *)v58 + 24LL) + 16LL),
              5,
              (__int64)(v61 - 9),
              3,
              v110);
            *(_QWORD *)(v62 + 56) = 0;
            sub_15F6500(v62, *(_QWORD *)(*(_QWORD *)v58 + 24LL), v58, v60, v59, (__int64)&v115, 0, 0, 0, 0);
          }
        }
        if ( v102 )
        {
          v63 = *(__int64 **)v98;
          v64 = (signed __int64 *)(*(_QWORD *)v98 + 8LL * *(unsigned int *)(v98 + 8));
          if ( *(signed __int64 **)v98 != v64 )
            goto LABEL_77;
        }
        else if ( v98 )
        {
          v63 = &v114;
          v64 = &v115;
          do
          {
LABEL_77:
            v65 = *v63++;
            sub_170BC50((__int64)v30, v65);
          }
          while ( v64 != v63 );
        }
        a2 = v103;
        v21 = sub_170BC50((__int64)v30, v103);
LABEL_12:
        if ( v99 )
        {
          sub_129E320(v99, a2);
          j_j___libc_free_0(v99, 464);
        }
        if ( v102 )
        {
          v22 = (unsigned __int64 *)(v100 & 0xFFFFFFFFFFFFFFF8LL);
          if ( (v100 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
          {
            if ( (unsigned __int64 *)*v22 != v22 + 2 )
              _libc_free(*v22);
            j_j___libc_free_0(v22, 48);
          }
        }
        v23 = v118;
        v24 = &v118[24 * (unsigned int)v119];
        if ( v118 != (_BYTE *)v24 )
        {
          do
          {
            v25 = *(v24 - 1);
            v24 -= 3;
            if ( v25 != -8 && v25 != 0 && v25 != -16 )
              sub_1649B30(v24);
          }
          while ( v23 != v24 );
          v24 = v118;
        }
        if ( v24 != (_QWORD *)v120 )
          _libc_free((unsigned __int64)v24);
        return v21;
      case 'N':
        v76 = *(v19 - 3);
        if ( !*(_BYTE *)(v76 + 16) && (*(_BYTE *)(v76 + 33) & 0x20) != 0 )
        {
          switch ( *(_DWORD *)(v76 + 36) )
          {
            case 0x71:
            case 0x72:
            case 0x74:
            case 0x75:
            case 0x90:
              goto LABEL_84;
            case 0x85:
            case 0x87:
            case 0x89:
              v77 = *((_DWORD *)v19 + 5) & 0xFFFFFFF;
              v78 = v19[3 * (3 - v77)];
              if ( *(_DWORD *)(v78 + 32) <= 0x40u )
              {
                v80 = *(_QWORD *)(v78 + 24) == 0;
              }
              else
              {
                v101 = *(_DWORD *)(v78 + 32);
                v95 = *((_DWORD *)v19 + 5) & 0xFFFFFFF;
                v79 = sub_16A57B0(v78 + 24);
                v77 = v95;
                v80 = v101 == v79;
              }
              if ( v80 )
              {
                v81 = v19[-3 * v77];
                if ( v107 == v81 )
                {
                  if ( v81 )
                    goto LABEL_84;
                }
              }
              goto LABEL_9;
            default:
              goto LABEL_9;
          }
        }
        a2 = (__int64)v104;
        if ( !sub_140B650((__int64)v19, v104) )
          goto LABEL_9;
LABEL_95:
        v71 = v119;
        if ( (unsigned int)v119 >= HIDWORD(v119) )
        {
          a2 = 0;
          sub_170B450((__int64)&v118, 0);
          v71 = v119;
        }
        v72 = &v118[24 * v71];
        if ( !v72 )
          goto LABEL_87;
        *v72 = 6;
        v72[1] = 0;
        v72[2] = v19;
        if ( v19 != (_QWORD *)-16LL && v19 != (_QWORD *)-8LL )
          goto LABEL_100;
        goto LABEL_101;
      default:
        goto LABEL_9;
    }
  }
}
