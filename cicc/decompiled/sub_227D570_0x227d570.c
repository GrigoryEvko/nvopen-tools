// Function: sub_227D570
// Address: 0x227d570
//
__int64 __fastcall sub_227D570(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // rbx
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 *v11; // rdx
  char v12; // al
  __int64 *v13; // rsi
  __int64 *v14; // rdx
  __int64 *v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r9
  int v19; // r8d
  __int64 *v20; // rdi
  __int64 v21; // r10
  unsigned int v22; // edx
  __int64 *v23; // rax
  __int64 v24; // r12
  int v25; // eax
  __int64 v26; // rax
  unsigned int v27; // ecx
  __int64 v28; // rsi
  __int64 *v29; // rsi
  __int64 *v30; // r14
  __int64 v31; // rdi
  __int64 (__fastcall *v32)(__int64, __int64, __int64, __int64); // rax
  char v33; // al
  __int64 v34; // rdx
  __int64 v35; // rcx
  int v36; // eax
  int v37; // ebx
  __int64 v38; // rax
  __int64 v39; // rsi
  __int64 v40; // rdx
  __int64 v41; // rcx
  int v42; // r9d
  unsigned int i; // eax
  __int64 v44; // rsi
  unsigned int v45; // eax
  __int64 v46; // r8
  char v47; // al
  __int64 *v48; // rbx
  __int64 *v49; // r13
  unsigned int v50; // ecx
  __int64 v51; // rdi
  unsigned int v52; // edx
  __int64 *v53; // rax
  __int64 v54; // r9
  int v55; // r8d
  int v56; // eax
  __int64 v57; // r9
  __int64 v58; // rdx
  unsigned int v59; // r10d
  unsigned int v60; // esi
  __int64 *v61; // rax
  __int64 v62; // r11
  int v63; // eax
  __int64 v64; // rbx
  __int64 v65; // rax
  __int64 v66; // r9
  __int64 v67; // r12
  char v68; // si
  __int64 v69; // rdi
  int v70; // edx
  __int64 v71; // rax
  __int64 v72; // r10
  __int64 v73; // rdx
  __int64 v74; // rax
  unsigned __int64 v75; // rdx
  __int64 v76; // r12
  __int64 *v77; // rbx
  __int64 v78; // r13
  __int64 *v79; // rdx
  __int64 **v80; // rax
  _QWORD *v81; // rax
  __int64 *v82; // rax
  __int64 v83; // rax
  __int64 v84; // rdx
  __int64 v85; // rcx
  int v86; // r10d
  unsigned int j; // eax
  _QWORD *v88; // rsi
  unsigned int v89; // eax
  __int64 v90; // rax
  __int64 v91; // rcx
  __int64 v92; // r8
  __int64 v93; // r9
  int v94; // eax
  int v95; // eax
  int v96; // r12d
  __int64 v97; // rax
  __int64 v98; // rax
  int v99; // eax
  __int64 *v100; // rsi
  __int64 *v101; // r14
  __int64 *v102; // rdi
  __int64 (__fastcall *v103)(__int64, __int64, void **, __int64 *); // rax
  char v104; // al
  __int64 v105; // rdx
  __int64 v106; // rcx
  __int64 v107; // rdx
  __int64 v108; // rcx
  int v109; // r8d
  __int64 v110; // [rsp+8h] [rbp-188h]
  unsigned __int64 v111; // [rsp+18h] [rbp-178h]
  __int64 v112; // [rsp+20h] [rbp-170h]
  __int64 *v113; // [rsp+28h] [rbp-168h]
  unsigned __int8 v114; // [rsp+3Eh] [rbp-152h]
  char v115; // [rsp+3Fh] [rbp-151h]
  __int64 v116; // [rsp+40h] [rbp-150h]
  __int64 *v119; // [rsp+60h] [rbp-130h]
  __int64 *v120; // [rsp+68h] [rbp-128h]
  __int64 *v121; // [rsp+68h] [rbp-128h]
  __int64 *v122; // [rsp+70h] [rbp-120h]
  __int64 *v124; // [rsp+88h] [rbp-108h]
  void *v125; // [rsp+90h] [rbp-100h] BYREF
  char v126[8]; // [rsp+98h] [rbp-F8h] BYREF
  __int64 *v127; // [rsp+A0h] [rbp-F0h] BYREF
  char v128[8]; // [rsp+A8h] [rbp-E8h] BYREF
  __int64 v129; // [rsp+B0h] [rbp-E0h]
  __int64 *v130[2]; // [rsp+C0h] [rbp-D0h] BYREF
  __int64 v131; // [rsp+D0h] [rbp-C0h]
  __int64 v132[20]; // [rsp+F0h] [rbp-A0h] BYREF

  if ( *(_DWORD *)(a3 + 72) == *(_DWORD *)(a3 + 68) )
  {
    v114 = 0;
    if ( (unsigned __int8)sub_B19060(a3, (__int64)&unk_4F82400, a3, a4) )
      return v114;
  }
  if ( (unsigned __int8)sub_B19060(a3 + 48, (__int64)&unk_4FDADC0, a3, a4)
    || !(unsigned __int8)sub_B19060(a3, (__int64)&unk_4F82400, v5, v6)
    && !(unsigned __int8)sub_B19060(a3, (__int64)&unk_4FDADC0, v9, v10)
    && !(unsigned __int8)sub_B19060(a3, (__int64)&unk_4F82400, v105, v106)
    && !(unsigned __int8)sub_B19060(a3, (__int64)&unk_4F82428, v107, v108) )
  {
LABEL_3:
    v7 = *a1;
    sub_227C140(*a1 + 64);
    sub_227BED0(v7 + 32);
    return 1;
  }
  sub_227B950(&v127, *(__int64 **)a4, (__int64)&unk_4F86C48);
  v11 = *(__int64 **)a4;
  if ( (*(_BYTE *)(*(_QWORD *)a4 + 8LL) & 1) != 0 )
  {
    if ( (__int64 *)v129 != v11 + 18 )
    {
LABEL_10:
      v12 = *(_BYTE *)(v129 + 8);
      goto LABEL_11;
    }
  }
  else if ( v129 != 16LL * *((unsigned int *)v11 + 6) + v11[2] )
  {
    goto LABEL_10;
  }
  v29 = *(__int64 **)(a4 + 8);
  v132[0] = (__int64)&unk_4F86C48;
  v132[1] = a2;
  sub_227BBF0(v130, v29, v132);
  v30 = *(__int64 **)a4;
  v31 = *(_QWORD *)(*(_QWORD *)(v131 + 16) + 24LL);
  v32 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v31 + 16LL);
  if ( v32 == sub_227A7A0 )
    v33 = sub_D23EA0(v31 + 8, a2, a3, a4);
  else
    v33 = v32(v31, a2, a3, a4);
  v126[0] = v33;
  v125 = &unk_4F86C48;
  sub_BBCF50((__int64)v132, (__int64)v30, (__int64 *)&v125, v126);
  v12 = *(_BYTE *)(v132[2] + 8);
LABEL_11:
  if ( v12 )
    goto LABEL_3;
  v13 = *(__int64 **)a4;
  sub_227B950(&v127, *(__int64 **)a4, (__int64)&unk_4F82418);
  v14 = *(__int64 **)a4;
  if ( (*(_BYTE *)(*(_QWORD *)a4 + 8LL) & 1) != 0 )
  {
    v15 = v14 + 2;
    v16 = 16;
  }
  else
  {
    v15 = (__int64 *)v14[2];
    v16 = 2LL * *((unsigned int *)v14 + 6);
  }
  if ( (__int64 *)v129 == &v15[v16] )
  {
    v100 = *(__int64 **)(a4 + 8);
    v132[0] = (__int64)&unk_4F82418;
    v132[1] = a2;
    sub_227BBF0(v130, v100, v132);
    v101 = *(__int64 **)a4;
    v102 = *(__int64 **)(*(_QWORD *)(v131 + 16) + 24LL);
    v103 = *(__int64 (__fastcall **)(__int64, __int64, void **, __int64 *))(*v102 + 16);
    v104 = v103 == sub_227A7B0
         ? sub_BBEB20(v102 + 1, a2, (void **)a3, (__int64 *)a4)
         : v103((__int64)v102, a2, (void **)a3, (__int64 *)a4);
    v13 = v101;
    v126[0] = v104;
    v125 = &unk_4F82418;
    sub_BBCF50((__int64)v132, (__int64)v101, (__int64 *)&v125, v126);
    v114 = *(_BYTE *)(v132[2] + 8);
  }
  else
  {
    v114 = *(_BYTE *)(v129 + 8);
  }
  if ( v114 )
    goto LABEL_3;
  v17 = *(unsigned int *)(a3 + 72);
  if ( *(_DWORD *)(a3 + 68) == (_DWORD)v17 )
  {
    v13 = (__int64 *)&unk_4F82400;
    v115 = sub_B19060(a3, (__int64)&unk_4F82400, v16 * 8, v17);
    if ( !v115 )
    {
      v13 = (__int64 *)&unk_4FDADC8;
      v115 = sub_B19060(a3, (__int64)&unk_4FDADC8, v34, v35);
    }
  }
  else
  {
    v115 = 0;
  }
  sub_D2AD40(a1[1], v13);
  v18 = a1[1];
  v19 = *(_DWORD *)(v18 + 440);
  v110 = v18;
  if ( v19 )
  {
    v20 = *(__int64 **)(v18 + 432);
    v21 = *v20;
    if ( *v20 )
    {
      while ( 1 )
      {
        v26 = *(unsigned int *)(v21 + 16);
        if ( (_DWORD)v26 )
          break;
        v27 = *(_DWORD *)(v18 + 600);
        v28 = *(_QWORD *)(v18 + 584);
        if ( v27 )
        {
          v22 = (v27 - 1) & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
          v23 = (__int64 *)(v28 + 16LL * v22);
          v24 = *v23;
          if ( v21 == *v23 )
            goto LABEL_23;
          v36 = 1;
          while ( v24 != -4096 )
          {
            v37 = v36 + 1;
            v38 = (v27 - 1) & (v22 + v36);
            v22 = v38;
            v23 = (__int64 *)(v28 + 16 * v38);
            v24 = *v23;
            if ( *v23 == v21 )
              goto LABEL_23;
            v36 = v37;
          }
        }
        v23 = (__int64 *)(v28 + 16LL * v27);
LABEL_23:
        v25 = *((_DWORD *)v23 + 2) + 1;
        if ( v25 != v19 )
        {
          v21 = v20[v25];
          if ( v21 )
            continue;
        }
        return v114;
      }
      v112 = v21;
      v124 = (__int64 *)a4;
LABEL_40:
      v122 = *(__int64 **)(v112 + 8);
      v113 = &v122[v26];
      while ( 1 )
      {
        v39 = *v122;
        memset(v132, 0, 0x68u);
        v116 = v39;
        v40 = *(unsigned int *)(*a1 + 88);
        v41 = *(_QWORD *)(*a1 + 72);
        if ( !(_DWORD)v40 )
          goto LABEL_55;
        v42 = 1;
        v111 = (unsigned __int64)(((unsigned int)&unk_4FDADB8 >> 9) ^ ((unsigned int)&unk_4FDADB8 >> 4)) << 32;
        for ( i = (v40 - 1)
                & (((0xBF58476D1CE4E5B9LL * (v111 | ((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4))) >> 31)
                 ^ (484763065 * (v111 | ((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4)))); ; i = (v40 - 1) & v45 )
        {
          v44 = v41 + 24LL * i;
          if ( *(_UNKNOWN **)v44 == &unk_4FDADB8 && v116 == *(_QWORD *)(v44 + 8) )
            break;
          if ( *(_QWORD *)v44 == -4096 && *(_QWORD *)(v44 + 8) == -4096 )
            goto LABEL_55;
          v45 = v42 + i;
          ++v42;
        }
        if ( v44 != v41 + 24 * v40 )
        {
          v46 = *(_QWORD *)(*(_QWORD *)(v44 + 16) + 24LL);
          if ( v46 )
            break;
        }
LABEL_55:
        if ( !v115 )
        {
          sub_227C930(*a1, v116, a3, v41);
          if ( LOBYTE(v132[12]) )
          {
LABEL_69:
            LOBYTE(v132[12]) = 0;
            sub_227AD40((__int64)v132);
          }
        }
LABEL_56:
        if ( v113 == ++v122 )
        {
          v50 = *(_DWORD *)(v110 + 600);
          v51 = *(_QWORD *)(v110 + 584);
          if ( !v50 )
            goto LABEL_148;
          v52 = (v50 - 1) & (((unsigned int)v112 >> 9) ^ ((unsigned int)v112 >> 4));
          v53 = (__int64 *)(v51 + 16LL * v52);
          v54 = *v53;
          if ( v112 != *v53 )
          {
            v99 = 1;
            while ( v54 != -4096 )
            {
              v109 = v99 + 1;
              v52 = (v50 - 1) & (v99 + v52);
              v53 = (__int64 *)(v51 + 16LL * v52);
              v54 = *v53;
              if ( *v53 == v112 )
                goto LABEL_59;
              v99 = v109;
            }
LABEL_148:
            v53 = (__int64 *)(v51 + 16LL * v50);
          }
LABEL_59:
          v55 = *(_DWORD *)(v110 + 440);
          v56 = *((_DWORD *)v53 + 2) + 1;
          if ( v56 == v55 )
            return v114;
          v57 = *(_QWORD *)(v110 + 432);
          v58 = *(_QWORD *)(v57 + 8LL * v56);
          if ( !v58 )
            return v114;
          v59 = v50 - 1;
          while ( 2 )
          {
            v26 = *(unsigned int *)(v58 + 16);
            if ( !(_DWORD)v26 )
            {
              if ( !v50 )
                goto LABEL_67;
              v60 = v59 & (((unsigned int)v58 >> 9) ^ ((unsigned int)v58 >> 4));
              v61 = (__int64 *)(v51 + 16LL * v60);
              v62 = *v61;
              if ( v58 != *v61 )
              {
                v95 = 1;
                while ( v62 != -4096 )
                {
                  v96 = v95 + 1;
                  v97 = v59 & (v60 + v95);
                  v60 = v97;
                  v61 = (__int64 *)(v51 + 16 * v97);
                  v62 = *v61;
                  if ( v58 == *v61 )
                    goto LABEL_63;
                  v95 = v96;
                }
LABEL_67:
                v61 = (__int64 *)(v51 + 16LL * v50);
              }
LABEL_63:
              v63 = *((_DWORD *)v61 + 2) + 1;
              if ( v55 == v63 )
                return v114;
              v58 = *(_QWORD *)(v57 + 8LL * v63);
              if ( !v58 )
                return v114;
              continue;
            }
            break;
          }
          v112 = v58;
          goto LABEL_40;
        }
      }
      v47 = *(_BYTE *)(v46 + 24) & 1;
      if ( *(_DWORD *)(v46 + 24) >> 1 )
      {
        if ( v47 )
        {
          v48 = (__int64 *)(v46 + 32);
          v49 = (__int64 *)(v46 + 64);
        }
        else
        {
          v48 = *(__int64 **)(v46 + 32);
          v46 = 16LL * *(unsigned int *)(v46 + 40);
          v49 = (__int64 *)((char *)v48 + v46);
          if ( v48 == (__int64 *)((char *)v48 + v46) )
            goto LABEL_55;
        }
        while ( *v48 == -4096 || *v48 == -8192 )
        {
          v48 += 2;
          if ( v48 == v49 )
            goto LABEL_55;
        }
      }
      else
      {
        if ( v47 )
        {
          v64 = v46 + 32;
          v65 = 32;
        }
        else
        {
          v64 = *(_QWORD *)(v46 + 32);
          v65 = 16LL * *(unsigned int *)(v46 + 40);
        }
        v48 = (__int64 *)(v65 + v64);
        v49 = v48;
      }
      if ( v48 == v49 )
        goto LABEL_55;
      while ( 1 )
      {
        v66 = *v48;
        v67 = *v124;
        v68 = *(_BYTE *)(*v124 + 8) & 1;
        if ( v68 )
        {
          v69 = v67 + 16;
          v70 = 7;
        }
        else
        {
          v74 = *(unsigned int *)(v67 + 24);
          v69 = *(_QWORD *)(v67 + 16);
          if ( !(_DWORD)v74 )
            goto LABEL_125;
          v70 = v74 - 1;
        }
        v41 = v70 & (((unsigned int)v66 >> 9) ^ ((unsigned int)v66 >> 4));
        v71 = v69 + 16 * v41;
        v72 = *(_QWORD *)v71;
        if ( v66 == *(_QWORD *)v71 )
          goto LABEL_77;
        v94 = 1;
        while ( v72 != -4096 )
        {
          v46 = (unsigned int)(v94 + 1);
          v98 = v70 & (unsigned int)(v41 + v94);
          v41 = (unsigned int)v98;
          v71 = v69 + 16 * v98;
          v72 = *(_QWORD *)v71;
          if ( v66 == *(_QWORD *)v71 )
            goto LABEL_77;
          v94 = v46;
        }
        if ( v68 )
        {
          v90 = 128;
          goto LABEL_126;
        }
        v74 = *(unsigned int *)(v67 + 24);
LABEL_125:
        v90 = 16 * v74;
LABEL_126:
        v71 = v69 + v90;
LABEL_77:
        v73 = 128;
        if ( !v68 )
          v73 = 16LL * *(unsigned int *)(v67 + 24);
        if ( v71 == v69 + v73 )
        {
          v83 = v124[1];
          v84 = *(unsigned int *)(v83 + 24);
          v85 = *(_QWORD *)(v83 + 8);
          if ( (_DWORD)v84 )
          {
            v86 = 1;
            for ( j = (v84 - 1)
                    & (((0xBF58476D1CE4E5B9LL
                       * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)
                        | ((unsigned __int64)(((unsigned int)v66 >> 9) ^ ((unsigned int)v66 >> 4)) << 32))) >> 31)
                     ^ (484763065 * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)))); ; j = (v84 - 1) & v89 )
            {
              v88 = (_QWORD *)(v85 + 24LL * j);
              if ( v66 == *v88 && a2 == v88[1] )
                break;
              if ( *v88 == -4096 && v88[1] == -4096 )
                goto LABEL_134;
              v89 = v86 + j;
              ++v86;
            }
          }
          else
          {
LABEL_134:
            v88 = (_QWORD *)(v85 + 24 * v84);
          }
          v121 = (__int64 *)*v48;
          v128[0] = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64 *))(**(_QWORD **)(v88[2] + 24LL) + 16LL))(
                      *(_QWORD *)(v88[2] + 24LL),
                      a2,
                      a3,
                      v124);
          v127 = v121;
          sub_BBCF50((__int64)v130, v67, (__int64 *)&v127, v128);
          v71 = v131;
        }
        if ( !*(_BYTE *)(v71 + 8) )
        {
          v48 += 2;
          goto LABEL_82;
        }
        if ( !LOBYTE(v132[12]) )
        {
          sub_C8CD80((__int64)v132, (__int64)&v132[4], a3, v41, v46, v66);
          sub_C8CD80((__int64)&v132[6], (__int64)&v132[10], a3 + 48, v91, v92, v93);
          LOBYTE(v132[12]) = 1;
        }
        v75 = v48[1] & 0xFFFFFFFFFFFFFFF8LL;
        if ( (v48[1] & 4) != 0 )
        {
          v41 = *(_QWORD *)v75;
          v48 += 2;
          v76 = *(_QWORD *)v75 + 8LL * *(unsigned int *)(v75 + 8);
        }
        else
        {
          v41 = (__int64)(v48 + 1);
          v48 += 2;
          if ( !v75 )
            goto LABEL_82;
          v76 = (__int64)v48;
        }
        if ( v41 != v76 )
        {
          v120 = v49;
          v119 = v48;
          v77 = (__int64 *)v41;
          while ( 1 )
          {
            v78 = *v77;
            if ( BYTE4(v132[3]) )
            {
              v46 = v132[1];
              v79 = (__int64 *)(v132[1] + 8LL * HIDWORD(v132[2]));
              v80 = (__int64 **)v132[1];
              if ( (__int64 *)v132[1] != v79 )
              {
                while ( (__int64 *)v78 != *v80 )
                {
                  if ( v79 == (__int64 *)++v80 )
                    goto LABEL_106;
                }
                --HIDWORD(v132[2]);
                v79 = *(__int64 **)(v132[1] + 8LL * HIDWORD(v132[2]));
                *v80 = v79;
                ++v132[0];
              }
            }
            else
            {
              v82 = sub_C8CA60((__int64)v132, v78);
              if ( v82 )
              {
                *v82 = -2;
                ++LODWORD(v132[3]);
                ++v132[0];
              }
            }
LABEL_106:
            if ( !BYTE4(v132[9]) )
              goto LABEL_113;
            v81 = (_QWORD *)v132[7];
            v79 = (__int64 *)(v132[7] + 8LL * HIDWORD(v132[8]));
            if ( (__int64 *)v132[7] != v79 )
            {
              while ( v78 != *v81 )
              {
                if ( v79 == ++v81 )
                  goto LABEL_114;
              }
              goto LABEL_111;
            }
LABEL_114:
            if ( HIDWORD(v132[8]) < LODWORD(v132[8]) )
            {
              ++HIDWORD(v132[8]);
              *v79 = v78;
              ++v132[6];
            }
            else
            {
LABEL_113:
              sub_C8CC70((__int64)&v132[6], v78, (__int64)v79, v41, v46, v66);
            }
LABEL_111:
            if ( (__int64 *)v76 == ++v77 )
            {
              v49 = v120;
              v48 = v119;
              break;
            }
          }
        }
LABEL_82:
        if ( v48 != v49 )
        {
          while ( *v48 == -4096 || *v48 == -8192 )
          {
            v48 += 2;
            if ( v49 == v48 )
              goto LABEL_86;
          }
          if ( v48 != v49 )
            continue;
        }
LABEL_86:
        if ( LOBYTE(v132[12]) )
        {
          sub_227C930(*a1, v116, (__int64)v132, v41);
          if ( LOBYTE(v132[12]) )
            goto LABEL_69;
          goto LABEL_56;
        }
        goto LABEL_55;
      }
    }
  }
  return v114;
}
