// Function: sub_20C08B0
// Address: 0x20c08b0
//
__int64 __fastcall sub_20C08B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r14
  unsigned __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 *v11; // rbx
  __int64 v12; // r12
  __int64 v13; // rax
  _BYTE *v14; // r13
  unsigned __int64 v15; // r12
  __int64 v16; // r14
  unsigned __int64 v17; // r15
  _QWORD *v18; // r14
  _QWORD *v19; // r13
  _QWORD *v20; // r12
  __int64 v21; // r13
  __int64 v22; // r12
  unsigned int v23; // eax
  int v24; // eax
  __int64 *v25; // rax
  __int64 v26; // r15
  unsigned __int64 v27; // rax
  __int64 v28; // rsi
  int v29; // r12d
  __int64 v30; // rdx
  __int64 *v31; // rax
  __int64 v32; // rdx
  __int64 *v33; // r13
  unsigned __int64 v34; // r12
  unsigned __int64 v35; // rbx
  __int64 v36; // rax
  unsigned __int64 v37; // r14
  _QWORD *v38; // r15
  unsigned __int64 v39; // r12
  _QWORD *v40; // rbx
  __int64 v41; // rsi
  __int64 v42; // r15
  __int64 v43; // r13
  __int64 v44; // r15
  unsigned int v45; // r14d
  unsigned __int64 v46; // rax
  __int64 v47; // rcx
  int v48; // ebx
  __int64 v49; // r12
  __int64 v50; // rax
  __int64 v51; // r15
  __int64 v52; // r13
  __int64 v53; // rsi
  __int64 v54; // rax
  __int64 v55; // rdi
  char v56; // al
  char v57; // di
  int v58; // eax
  __int64 v59; // rax
  __int64 v60; // r12
  __int64 v61; // rbx
  unsigned __int64 v62; // rax
  __int64 *v63; // r9
  __int64 v64; // rbx
  __int64 v65; // r10
  __int64 v66; // r13
  __int64 v67; // rax
  __int64 v68; // r15
  __int64 v69; // rdx
  __int64 v70; // rdx
  __int64 v72; // rax
  __int64 v73; // rdx
  char v74; // al
  bool v75; // zf
  __int64 v76; // rax
  __int64 v77; // rax
  unsigned int v78; // eax
  __int64 v79; // rax
  __int64 v80; // rax
  __int64 v81; // r14
  __int64 v82; // rax
  unsigned int v85; // [rsp+1Ch] [rbp-444h]
  unsigned int v87; // [rsp+20h] [rbp-440h]
  _DWORD *v88; // [rsp+28h] [rbp-438h]
  int v89; // [rsp+28h] [rbp-438h]
  unsigned int v90; // [rsp+30h] [rbp-430h]
  __int64 v91; // [rsp+30h] [rbp-430h]
  __int64 *v92; // [rsp+38h] [rbp-428h]
  bool v93; // [rsp+40h] [rbp-420h]
  int v94; // [rsp+40h] [rbp-420h]
  __int64 v95; // [rsp+40h] [rbp-420h]
  __int64 v96; // [rsp+40h] [rbp-420h]
  unsigned int v97; // [rsp+48h] [rbp-418h]
  __int64 *v98; // [rsp+48h] [rbp-418h]
  __int64 *v99; // [rsp+50h] [rbp-410h]
  __int64 v100; // [rsp+50h] [rbp-410h]
  __int64 v101; // [rsp+50h] [rbp-410h]
  unsigned __int64 v102; // [rsp+50h] [rbp-410h]
  int v104; // [rsp+60h] [rbp-400h]
  int v105; // [rsp+64h] [rbp-3FCh]
  char v106; // [rsp+68h] [rbp-3F8h]
  char v107; // [rsp+69h] [rbp-3F7h]
  char v108; // [rsp+6Ah] [rbp-3F6h]
  char v109; // [rsp+6Bh] [rbp-3F5h]
  int v110; // [rsp+6Ch] [rbp-3F4h]
  _BYTE *v111; // [rsp+70h] [rbp-3F0h] BYREF
  __int64 v112; // [rsp+78h] [rbp-3E8h]
  _BYTE v113[32]; // [rsp+80h] [rbp-3E0h] BYREF
  _BYTE *v114; // [rsp+A0h] [rbp-3C0h] BYREF
  __int64 v115; // [rsp+A8h] [rbp-3B8h]
  _BYTE v116[112]; // [rsp+B0h] [rbp-3B0h] BYREF
  __int64 *v117; // [rsp+120h] [rbp-340h] BYREF
  unsigned int v118; // [rsp+128h] [rbp-338h]
  _BYTE v119[816]; // [rsp+130h] [rbp-330h] BYREF

  v5 = a2;
  v88 = (_DWORD *)(a5 & 0xFFFFFFFFFFFFFFF8LL);
  v6 = (a5 & 0xFFFFFFFFFFFFFFF8LL) - 72;
  if ( (a5 & 4) != 0 )
    v6 = (a5 & 0xFFFFFFFFFFFFFFF8LL) - 24;
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  sub_15F1410(&v117, *(_BYTE **)(*(_QWORD *)v6 + 56LL), *(_QWORD *)(*(_QWORD *)v6 + 64LL));
  v11 = v117;
  v92 = &v117[24 * v118];
  if ( v117 != v92 )
  {
    v85 = 0;
    v90 = 0;
    v97 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        v12 = *(_QWORD *)(a1 + 8);
        if ( v12 == *(_QWORD *)(a1 + 16) )
        {
          sub_20C00F0((__int64 *)a1, *(int **)(a1 + 8), v11);
          v22 = *(_QWORD *)(a1 + 8);
          v21 = v22 - 248;
        }
        else
        {
          v104 = *(_DWORD *)v11;
          v105 = *((_DWORD *)v11 + 1);
          v106 = *((_BYTE *)v11 + 8);
          v107 = *((_BYTE *)v11 + 9);
          v108 = *((_BYTE *)v11 + 10);
          v109 = *((_BYTE *)v11 + 11);
          v110 = *((_DWORD *)v11 + 3);
          v111 = v113;
          v112 = 0x100000000LL;
          if ( *((_DWORD *)v11 + 6) )
            sub_20BDAA0((__int64)&v111, (__int64)(v11 + 2), v7, v8);
          v114 = v116;
          v115 = 0x200000000LL;
          if ( *((_DWORD *)v11 + 18) )
            sub_20BFC20((__int64)&v114, (__int64)(v11 + 8));
          if ( v12 )
          {
            *(_DWORD *)v12 = v104;
            *(_DWORD *)(v12 + 4) = v105;
            *(_BYTE *)(v12 + 8) = v106;
            *(_BYTE *)(v12 + 9) = v107;
            *(_BYTE *)(v12 + 10) = v108;
            *(_BYTE *)(v12 + 11) = v109;
            *(_DWORD *)(v12 + 12) = v110;
            *(_QWORD *)(v12 + 16) = v12 + 32;
            *(_QWORD *)(v12 + 24) = 0x100000000LL;
            if ( (_DWORD)v112 )
              sub_20BDAA0(v12 + 16, (__int64)&v111, v7, (unsigned int)v112);
            *(_QWORD *)(v12 + 64) = v12 + 80;
            *(_QWORD *)(v12 + 72) = 0x200000000LL;
            v13 = (unsigned int)v115;
            if ( (_DWORD)v115 )
            {
              sub_20BFC20(v12 + 64, (__int64)&v114);
              v13 = (unsigned int)v115;
            }
            *(_QWORD *)(v12 + 200) = 0;
            *(_QWORD *)(v12 + 192) = v12 + 208;
            *(_BYTE *)(v12 + 208) = 0;
            *(_DWORD *)(v12 + 224) = 4;
            *(_QWORD *)(v12 + 232) = 0;
            *(_BYTE *)(v12 + 240) = 1;
          }
          else
          {
            v13 = (unsigned int)v115;
          }
          v14 = v114;
          v7 = 7 * v13;
          v15 = (unsigned __int64)&v114[56 * v13];
          if ( v114 != (_BYTE *)v15 )
          {
            do
            {
              v16 = *(unsigned int *)(v15 - 40);
              v17 = *(_QWORD *)(v15 - 48);
              v15 -= 56LL;
              v18 = (_QWORD *)(v17 + 32 * v16);
              if ( (_QWORD *)v17 != v18 )
              {
                do
                {
                  v18 -= 4;
                  if ( (_QWORD *)*v18 != v18 + 2 )
                    j_j___libc_free_0(*v18, v18[2] + 1LL);
                }
                while ( (_QWORD *)v17 != v18 );
                v17 = *(_QWORD *)(v15 + 8);
              }
              if ( v17 != v15 + 24 )
                _libc_free(v17);
            }
            while ( v14 != (_BYTE *)v15 );
            v15 = (unsigned __int64)v114;
          }
          if ( (_BYTE *)v15 != v116 )
            _libc_free(v15);
          v19 = v111;
          v20 = &v111[32 * (unsigned int)v112];
          if ( v111 != (_BYTE *)v20 )
          {
            do
            {
              v20 -= 4;
              if ( (_QWORD *)*v20 != v20 + 2 )
                j_j___libc_free_0(*v20, v20[2] + 1LL);
            }
            while ( v19 != v20 );
            v20 = v111;
          }
          if ( v20 != (_QWORD *)v113 )
            _libc_free((unsigned __int64)v20);
          v21 = *(_QWORD *)(a1 + 8);
          v22 = v21 + 248;
          *(_QWORD *)(a1 + 8) = v21 + 248;
        }
        v8 = v97;
        v23 = *(_DWORD *)(v22 - 176);
        *(_BYTE *)(v22 - 8) = 1;
        if ( v97 >= v23 )
          v23 = v97;
        v97 = v23;
        v24 = *(_DWORD *)(v22 - 248);
        if ( v24 )
          break;
        v72 = v90;
        v8 = v90 + 1;
        v7 = v88[5] & 0xFFFFFFF;
        ++v90;
        v31 = *(__int64 **)&v88[6 * (v72 - v7)];
        *(_QWORD *)(v22 - 16) = v31;
LABEL_53:
        if ( !v31 )
          goto LABEL_58;
        v26 = *v31;
        if ( !*(_BYTE *)(v22 - 238) )
        {
          v27 = *(unsigned __int8 *)(v26 + 8);
          if ( (_BYTE)v27 != 13 )
            goto LABEL_56;
          goto LABEL_45;
        }
LABEL_43:
        if ( *(_BYTE *)(v26 + 8) != 15 )
          sub_16BD130("Indirect operand for inline asm not a pointer!", 1u);
        v26 = *(_QWORD *)(v26 + 24);
        v27 = *(unsigned __int8 *)(v26 + 8);
        if ( (_BYTE)v27 != 13 )
          goto LABEL_56;
LABEL_45:
        if ( *(_DWORD *)(v26 + 12) != 1 )
          goto LABEL_46;
        v26 = **(_QWORD **)(v26 + 16);
        v27 = *(unsigned __int8 *)(v26 + 8);
LABEL_56:
        if ( (unsigned __int8)v27 > 0x10u )
          goto LABEL_57;
        v73 = 100990;
        if ( _bittest64(&v73, v27) )
          goto LABEL_120;
        if ( (_BYTE)v27 == 16 )
        {
LABEL_134:
          if ( !sub_16435F0(v26, 0) )
          {
            LOBYTE(v27) = *(_BYTE *)(v26 + 8);
            goto LABEL_120;
          }
          v27 = *(unsigned __int8 *)(v26 + 8);
LABEL_47:
          v28 = v26;
          v29 = 1;
          while ( 2 )
          {
            switch ( v27 )
            {
              case 0uLL:
              case 8uLL:
              case 0xAuLL:
              case 0xCuLL:
              case 0x10uLL:
                v80 = *(_QWORD *)(v28 + 32);
                v28 = *(_QWORD *)(v28 + 24);
                v29 *= (_DWORD)v80;
                v27 = *(unsigned __int8 *)(v28 + 8);
                continue;
              case 1uLL:
                LODWORD(v77) = 16;
                break;
              case 2uLL:
                LODWORD(v77) = 32;
                break;
              case 3uLL:
              case 9uLL:
                LODWORD(v77) = 64;
                break;
              case 4uLL:
                LODWORD(v77) = 80;
                break;
              case 5uLL:
              case 6uLL:
                LODWORD(v77) = 128;
                break;
              case 7uLL:
                LODWORD(v77) = 8 * sub_15A9520(a3, 0);
                break;
              case 0xBuLL:
                LODWORD(v77) = *(_DWORD *)(v28 + 8) >> 8;
                break;
              case 0xDuLL:
                v77 = 8LL * *(_QWORD *)sub_15A9930(a3, v28);
                break;
              case 0xEuLL:
                v81 = *(_QWORD *)(v28 + 32);
                v96 = *(_QWORD *)(v28 + 24);
                v102 = (unsigned int)sub_15A9FE0(a3, v96);
                v82 = sub_127FA20(a3, v96);
                v7 = (v102 + ((unsigned __int64)(v82 + 7) >> 3) - 1) % v102;
                v8 = v81 * v102;
                v77 = 8 * v81 * v102 * ((v102 + ((unsigned __int64)(v82 + 7) >> 3) - 1) / v102);
                break;
              case 0xFuLL:
                LODWORD(v77) = 8 * sub_15A9520(a3, *(_DWORD *)(v28 + 8) >> 8);
                break;
            }
            break;
          }
          v78 = v29 * v77;
          if ( v78 > 0x40 )
          {
            if ( v78 == 128 )
            {
LABEL_148:
              v79 = sub_1644900(*(_QWORD **)v26, v78);
              *(_BYTE *)(v21 + 240) = sub_1F59410(v79);
              goto LABEL_58;
            }
            v11 += 24;
            if ( v92 == v11 )
              goto LABEL_59;
          }
          else if ( v78 > 7 )
          {
            v8 = 0x100000001000101LL;
            v7 = v78 - 8;
            if ( _bittest64(&v8, v7) )
              goto LABEL_148;
            v11 += 24;
            if ( v92 == v11 )
              goto LABEL_59;
          }
          else
          {
            if ( v78 == 1 )
              goto LABEL_148;
LABEL_58:
            v11 += 24;
            if ( v92 == v11 )
              goto LABEL_59;
          }
        }
        else
        {
LABEL_46:
          v8 = 35454;
          v7 = (unsigned __int8)v27;
          if ( _bittest64(&v8, v27) )
            goto LABEL_47;
          if ( (unsigned int)(unsigned __int8)v27 - 13 <= 1 )
            goto LABEL_134;
LABEL_120:
          if ( (_BYTE)v27 != 15 )
          {
LABEL_57:
            *(_BYTE *)(v21 + 240) = sub_1F59410(v26);
            goto LABEL_58;
          }
          v7 = 8 * (unsigned int)sub_15A9520(a3, *(_DWORD *)(v26 + 8) >> 8);
          if ( (_DWORD)v7 == 32 )
          {
            v74 = 5;
          }
          else if ( (unsigned int)v7 > 0x20 )
          {
            v74 = 6;
            if ( (_DWORD)v7 != 64 )
            {
              v75 = (_DWORD)v7 == 128;
              v74 = 0;
              v7 = 7;
              if ( v75 )
                v74 = 7;
            }
          }
          else
          {
            v74 = 3;
            if ( (_DWORD)v7 != 8 )
              v74 = 4 * ((_DWORD)v7 == 16);
          }
          *(_BYTE *)(v22 - 8) = v74;
          v11 += 24;
          if ( v92 == v11 )
          {
LABEL_59:
            v32 = v97;
            v5 = a2;
            v99 = v117;
            v92 = &v117[24 * v118];
            v93 = v97 != 0;
            if ( v117 != v92 )
            {
              v33 = &v117[24 * v118];
              do
              {
                v32 = *((unsigned int *)v33 - 30);
                v34 = *(v33 - 16);
                v33 -= 24;
                v35 = v34 + 56 * v32;
                if ( v34 != v35 )
                {
                  do
                  {
                    v36 = *(unsigned int *)(v35 - 40);
                    v37 = *(_QWORD *)(v35 - 48);
                    v35 -= 56LL;
                    v36 *= 32;
                    v38 = (_QWORD *)(v37 + v36);
                    if ( v37 != v37 + v36 )
                    {
                      do
                      {
                        v38 -= 4;
                        v32 = (__int64)(v38 + 2);
                        if ( (_QWORD *)*v38 != v38 + 2 )
                          j_j___libc_free_0(*v38, v38[2] + 1LL);
                      }
                      while ( (_QWORD *)v37 != v38 );
                      v37 = *(_QWORD *)(v35 + 8);
                    }
                    if ( v37 != v35 + 24 )
                      _libc_free(v37);
                  }
                  while ( v34 != v35 );
                  v34 = v33[8];
                }
                if ( (__int64 *)v34 != v33 + 10 )
                  _libc_free(v34);
                v39 = v33[2];
                v40 = (_QWORD *)(v39 + 32LL * *((unsigned int *)v33 + 6));
                if ( (_QWORD *)v39 != v40 )
                {
                  do
                  {
                    v40 -= 4;
                    if ( (_QWORD *)*v40 != v40 + 2 )
                      j_j___libc_free_0(*v40, v40[2] + 1LL);
                  }
                  while ( (_QWORD *)v39 != v40 );
                  v39 = v33[2];
                }
                if ( (__int64 *)v39 != v33 + 4 )
                  _libc_free(v39);
              }
              while ( v99 != v33 );
              v5 = a2;
              v92 = v117;
            }
            if ( v92 != (__int64 *)v119 )
              goto LABEL_83;
            goto LABEL_84;
          }
        }
      }
      if ( v24 != 1 )
        goto LABEL_52;
      if ( !*(_BYTE *)(v22 - 238) )
        break;
      v8 = v90 + 1;
      v7 = v88[5] & 0xFFFFFFF;
      v25 = *(__int64 **)&v88[6 * (v90 - v7)];
      *(_QWORD *)(v22 - 16) = v25;
      if ( v25 )
      {
        ++v90;
        v26 = *v25;
        goto LABEL_43;
      }
      ++v90;
      v11 += 24;
      if ( v92 == v11 )
        goto LABEL_59;
    }
    v30 = *(_QWORD *)v88;
    if ( *(_BYTE *)(*(_QWORD *)v88 + 8LL) == 13 )
      v30 = *(_QWORD *)(*(_QWORD *)(v30 + 16) + 8LL * v85);
    ++v85;
    *(_BYTE *)(v22 - 8) = sub_204D4D0(a2, a3, v30);
LABEL_52:
    v31 = *(__int64 **)(v22 - 16);
    goto LABEL_53;
  }
  if ( v117 == (__int64 *)v119 )
  {
    v41 = *(_QWORD *)(a1 + 8);
    v42 = *(_QWORD *)a1;
    goto LABEL_107;
  }
  v93 = 0;
  v97 = 0;
LABEL_83:
  _libc_free((unsigned __int64)v92);
LABEL_84:
  v41 = *(_QWORD *)(a1 + 8);
  v42 = *(_QWORD *)a1;
  if ( v41 == *(_QWORD *)a1 || !v93 )
    goto LABEL_107;
  v89 = -1;
  v43 = *(_QWORD *)a1;
  v44 = v5;
  v87 = 0;
  v45 = 0;
  do
  {
    v46 = 0xEF7BDEF7BDEF7BDFLL * ((v41 - v43) >> 3);
    v47 = (unsigned int)v46;
    if ( !(_DWORD)v46 )
    {
      v48 = 0;
      goto LABEL_98;
    }
    v48 = 0;
    v49 = 0;
    v100 = 248LL * (unsigned int)v46;
    v50 = v44;
    v51 = v43;
    v52 = v50;
    while ( 1 )
    {
      v53 = v51 + v49;
      if ( *(_DWORD *)(v51 + v49) == 2 )
        goto LABEL_96;
      v54 = *(int *)(v53 + 4);
      if ( (_DWORD)v54 != -1 )
      {
        v55 = 31 * v54;
        v56 = *(_BYTE *)(v53 + 240);
        v10 = v51 + 8 * v55;
        v57 = *(_BYTE *)(v10 + 240);
        if ( v56 != v57 )
        {
          if ( ((unsigned __int8)(v57 - 14) <= 0x47u || (unsigned __int8)(v57 - 2) <= 5u) != ((unsigned __int8)(v56 - 2) <= 5u
                                                                                           || (unsigned __int8)(v56 - 14) <= 0x47u) )
            break;
          v91 = v10;
          v94 = sub_1F3E310((_BYTE *)(v53 + 240));
          v53 = v51 + v49;
          if ( v94 != (unsigned int)sub_1F3E310((_BYTE *)(v91 + 240)) )
            break;
        }
      }
      v58 = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD, __int64))(*(_QWORD *)v52 + 1360LL))(v52, v53, v45, v47);
      if ( v58 == -1 )
      {
        v44 = v52;
        v41 = *(_QWORD *)(a1 + 8);
        v43 = *(_QWORD *)a1;
        goto LABEL_139;
      }
      v48 += v58;
      v51 = *(_QWORD *)a1;
LABEL_96:
      v49 += 248;
      if ( v100 == v49 )
      {
        v59 = v52;
        v43 = v51;
        v44 = v59;
        v41 = *(_QWORD *)(a1 + 8);
        v47 = -1108378657 * (unsigned int)((v41 - v43) >> 3);
        goto LABEL_98;
      }
    }
    v76 = v52;
    v43 = v51;
    v44 = v76;
    v41 = *(_QWORD *)(a1 + 8);
LABEL_139:
    v48 = -1;
    v47 = -1108378657 * (unsigned int)((v41 - v43) >> 3);
LABEL_98:
    if ( v48 > v89 )
    {
      v89 = v48;
      v87 = v45;
    }
    ++v45;
  }
  while ( v45 < v97 );
  v5 = v44;
  v42 = v43;
  if ( (_DWORD)v47 )
  {
    v60 = 0;
    v61 = 248 * v47;
    do
    {
      if ( *(_DWORD *)(v42 + v60) != 2 )
      {
        sub_15EB7E0(v42 + v60, v87, v32, v47, v9, v10);
        v42 = *(_QWORD *)a1;
      }
      v60 += 248;
    }
    while ( v61 != v60 );
    v41 = *(_QWORD *)(a1 + 8);
LABEL_107:
    v62 = 0xEF7BDEF7BDEF7BDFLL * ((v41 - v42) >> 3);
    if ( (_DWORD)v62 )
    {
      v63 = (__int64 *)a1;
      v64 = 0;
      v65 = 248LL * (unsigned int)(v62 - 1);
      while ( 1 )
      {
        v66 = v42 + v64;
        v67 = *(int *)(v42 + v64 + 4);
        if ( (_DWORD)v67 != -1 )
        {
          v68 = v42 + 248 * v67;
          if ( *(_BYTE *)(v66 + 240) != *(_BYTE *)(v68 + 240) )
          {
            v95 = v65;
            v98 = v63;
            (*(void (__fastcall **)(__int64, __int64, _QWORD, _QWORD))(*(_QWORD *)v5 + 1392LL))(
              v5,
              a4,
              *(_QWORD *)(v66 + 192),
              *(_QWORD *)(v66 + 200));
            v101 = v69;
            (*(void (__fastcall **)(__int64, __int64, _QWORD, _QWORD, _QWORD))(*(_QWORD *)v5 + 1392LL))(
              v5,
              a4,
              *(_QWORD *)(v68 + 192),
              *(_QWORD *)(v68 + 200),
              *(unsigned __int8 *)(v68 + 240));
            if ( ((unsigned __int8)(*(_BYTE *)(v68 + 240) - 14) <= 0x47u
               || (unsigned __int8)(*(_BYTE *)(v68 + 240) - 2) <= 5u) != ((unsigned __int8)(*(_BYTE *)(v66 + 240) - 2) <= 5u
                                                                       || (unsigned __int8)(*(_BYTE *)(v66 + 240) - 14) <= 0x47u)
              || (v63 = v98, v65 = v95, v101 != v70) )
            {
              sub_16BD130(
                "Unsupported asm: input constraint with a matching output constraint of incompatible type!",
                1u);
            }
          }
        }
        if ( v65 == v64 )
          break;
        v42 = *v63;
        v64 += 248;
      }
    }
  }
  return a1;
}
