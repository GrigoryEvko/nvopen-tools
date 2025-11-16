// Function: sub_199BED0
// Address: 0x199bed0
//
void __fastcall sub_199BED0(__int64 *a1)
{
  __int64 *v1; // r14
  __int64 v2; // rdi
  __int64 v3; // rbx
  __int64 *v4; // rax
  __int64 v5; // r13
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rdi
  int v9; // r8d
  __int64 v10; // rsi
  __int64 v11; // rax
  unsigned int v12; // r9d
  __int64 *v13; // rax
  __int64 v14; // rcx
  __int64 *v15; // rbx
  __int64 v16; // rax
  __int64 i; // r12
  __int64 v18; // rax
  __int64 v19; // rbx
  __int64 v20; // r15
  __int64 v21; // r12
  _QWORD *v22; // rdx
  _QWORD *v23; // rax
  _QWORD *v24; // r13
  __int64 v25; // rax
  __int64 **v26; // r12
  __int64 v27; // rax
  __int64 v28; // r12
  __int64 v29; // rbx
  __int64 v30; // r13
  _QWORD *v31; // rax
  __int64 v32; // rax
  _BYTE *v33; // r15
  _QWORD *v34; // rdx
  __int64 **v35; // rdi
  __int64 **v36; // rax
  __int64 *v37; // rbx
  __int64 **v38; // r14
  char v39; // dl
  __int64 v40; // r13
  __int64 *v41; // rax
  __int64 *v42; // rsi
  __int64 *v43; // rcx
  __int64 v44; // rdx
  __int64 v45; // rsi
  __int64 v46; // rdx
  __int64 v47; // rcx
  __int64 v48; // rax
  __int64 v49; // rdx
  __int64 v50; // rbx
  __int64 v51; // r13
  __int64 v52; // r8
  char v53; // di
  unsigned int v54; // esi
  __int64 v55; // rdx
  __int64 v56; // rax
  __int64 v57; // rcx
  __int64 v58; // rax
  __int64 v59; // rdx
  __int64 v60; // rdx
  __int64 v61; // rax
  __int64 v62; // r15
  unsigned __int64 v63; // r12
  unsigned int *v64; // rbx
  __int64 v65; // rax
  __int64 v66; // rcx
  __int64 v67; // rax
  __int64 v68; // rsi
  __int64 v69; // r13
  __int64 v70; // rbx
  __int64 v71; // r12
  int v72; // r9d
  __int64 v73; // rax
  __int64 v74; // rax
  __int64 v75; // r8
  __int64 v76; // r13
  __int64 v77; // rbx
  __int64 v78; // rdx
  unsigned __int64 v79; // rax
  __int64 v80; // rcx
  const void *v81; // rsi
  __int64 v82; // rax
  __int64 *v83; // r13
  __int64 v84; // rbx
  _QWORD *v85; // rdi
  __int64 v86; // rsi
  _QWORD *v87; // rsi
  _QWORD *v88; // rax
  __int64 v89; // rax
  _QWORD *v90; // rdi
  unsigned int v91; // r8d
  _QWORD *v92; // rcx
  _QWORD *v93; // rdx
  unsigned __int64 v94; // rax
  unsigned __int64 *v95; // rbx
  __int64 v96; // r12
  _BYTE *v97; // rbx
  unsigned __int64 v98; // r12
  unsigned __int64 v99; // rdi
  unsigned __int64 v100; // rdi
  __int64 v101; // rax
  __int64 v102; // rdx
  __int64 v103; // rax
  int v104; // eax
  int v105; // r10d
  unsigned __int64 v106; // [rsp+10h] [rbp-5A0h]
  unsigned __int64 v107; // [rsp+10h] [rbp-5A0h]
  int v108; // [rsp+18h] [rbp-598h]
  int v109; // [rsp+20h] [rbp-590h]
  __int64 **v110; // [rsp+28h] [rbp-588h]
  __int64 v111; // [rsp+28h] [rbp-588h]
  unsigned int v112; // [rsp+28h] [rbp-588h]
  __int64 v113; // [rsp+28h] [rbp-588h]
  _BYTE *v114; // [rsp+30h] [rbp-580h]
  _BOOL4 v115; // [rsp+30h] [rbp-580h]
  unsigned int v116; // [rsp+30h] [rbp-580h]
  int v117; // [rsp+38h] [rbp-578h]
  __int64 v118; // [rsp+40h] [rbp-570h]
  int v119; // [rsp+40h] [rbp-570h]
  __int64 v120; // [rsp+48h] [rbp-568h]
  int v121; // [rsp+48h] [rbp-568h]
  __int64 v122; // [rsp+50h] [rbp-560h] BYREF
  __int64 *v123; // [rsp+58h] [rbp-558h]
  __int64 *v124; // [rsp+60h] [rbp-550h]
  __int64 v125; // [rsp+68h] [rbp-548h]
  int v126; // [rsp+70h] [rbp-540h]
  _BYTE v127[40]; // [rsp+78h] [rbp-538h] BYREF
  _BYTE *v128; // [rsp+A0h] [rbp-510h] BYREF
  __int64 v129; // [rsp+A8h] [rbp-508h]
  _BYTE v130[64]; // [rsp+B0h] [rbp-500h] BYREF
  _BYTE *v131; // [rsp+F0h] [rbp-4C0h] BYREF
  __int64 v132; // [rsp+F8h] [rbp-4B8h]
  _BYTE v133[1200]; // [rsp+100h] [rbp-4B0h] BYREF

  v1 = a1;
  v2 = a1[5];
  v3 = v1[2];
  v131 = v133;
  v132 = 0x800000000LL;
  v129 = 0x800000000LL;
  v4 = *(__int64 **)(v2 + 32);
  v128 = v130;
  v5 = *v4;
  v6 = sub_13FCB50(v2);
  v7 = *(unsigned int *)(v3 + 48);
  if ( !(_DWORD)v7 )
    goto LABEL_177;
  v8 = v6;
  v9 = v7 - 1;
  v10 = *(_QWORD *)(v3 + 32);
  v11 = ((_DWORD)v7 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
  v12 = v11;
  v13 = (__int64 *)(v10 + 16 * v11);
  v14 = *v13;
  if ( v8 != *v13 )
  {
    v104 = 1;
    while ( v14 != -8 )
    {
      v105 = v104 + 1;
      v12 = v9 & (v104 + v12);
      v13 = (__int64 *)(v10 + 16LL * v12);
      v14 = *v13;
      if ( v8 == *v13 )
        goto LABEL_3;
      v104 = v105;
    }
LABEL_177:
    BUG();
  }
LABEL_3:
  if ( v13 == (__int64 *)(v10 + 16 * v7) )
    goto LABEL_177;
  v15 = (__int64 *)v13[1];
  v16 = (unsigned int)v129;
  for ( i = *v15; v5 != *v15; i = *v15 )
  {
    if ( HIDWORD(v129) <= (unsigned int)v16 )
    {
      sub_16CD150((__int64)&v128, v130, 0, 8, v9, v12);
      v16 = (unsigned int)v129;
    }
    *(_QWORD *)&v128[8 * v16] = i;
    v16 = (unsigned int)(v129 + 1);
    LODWORD(v129) = v129 + 1;
    v15 = (__int64 *)v15[1];
  }
  if ( HIDWORD(v129) <= (unsigned int)v16 )
  {
    sub_16CD150((__int64)&v128, v130, 0, 8, v9, v12);
    v16 = (unsigned int)v129;
  }
  *(_QWORD *)&v128[8 * v16] = v5;
  LODWORD(v129) = v129 + 1;
  v106 = (unsigned __int64)v128;
  v114 = &v128[8 * (unsigned int)v129];
  if ( v128 != v114 )
  {
    while ( 1 )
    {
      v18 = *((_QWORD *)v114 - 1);
      v19 = *(_QWORD *)(v18 + 48);
      v120 = v18 + 40;
      if ( v18 + 40 != v19 )
        break;
LABEL_72:
      v114 -= 8;
      if ( (_BYTE *)v106 == v114 )
        goto LABEL_73;
    }
    while ( 1 )
    {
      if ( !v19 )
        BUG();
      v20 = v19 - 24;
      if ( *(_BYTE *)(v19 - 8) == 77 )
        goto LABEL_14;
      v21 = *v1;
      v22 = *(_QWORD **)(*v1 + 56);
      v23 = *(_QWORD **)(*v1 + 48);
      if ( v22 == v23 )
        break;
      v24 = &v22[*(unsigned int *)(v21 + 64)];
      v23 = sub_16CC9F0(v21 + 40, v19 - 24);
      if ( v20 == *v23 )
      {
        v46 = *(_QWORD *)(v21 + 56);
        if ( v46 == *(_QWORD *)(v21 + 48) )
          v47 = *(unsigned int *)(v21 + 68);
        else
          v47 = *(unsigned int *)(v21 + 64);
        v93 = (_QWORD *)(v46 + 8 * v47);
        goto LABEL_61;
      }
      v25 = *(_QWORD *)(v21 + 56);
      if ( v25 == *(_QWORD *)(v21 + 48) )
      {
        v23 = (_QWORD *)(v25 + 8LL * *(unsigned int *)(v21 + 68));
        v93 = v23;
        goto LABEL_61;
      }
      v23 = (_QWORD *)(v25 + 8LL * *(unsigned int *)(v21 + 64));
LABEL_21:
      if ( v23 == v24 )
        goto LABEL_14;
      v26 = (__int64 **)(v19 - 24);
      if ( sub_1456C80(v1[1], *(_QWORD *)(v19 - 24)) && *(_WORD *)(sub_146F1B0(v1[1], v19 - 24) + 24) != 10 )
        goto LABEL_14;
      v27 = *((unsigned int *)v1 + 8078);
      if ( !(_DWORD)v27 )
        goto LABEL_36;
      v118 = v19;
      v110 = (__int64 **)(v19 - 24);
      v28 = v19 - 24;
      v29 = 0;
      v30 = 144 * v27;
      do
      {
        while ( 1 )
        {
          v33 = &v131[v29];
          v31 = *(_QWORD **)&v131[v29 + 80];
          if ( *(_QWORD **)&v131[v29 + 88] == v31 )
            break;
          v31 = sub_16CC9F0((__int64)(v33 + 72), v28);
          if ( v28 == *v31 )
          {
            v44 = *((_QWORD *)v33 + 11);
            if ( v44 == *((_QWORD *)v33 + 10) )
              v45 = *((unsigned int *)v33 + 25);
            else
              v45 = *((unsigned int *)v33 + 24);
            v34 = (_QWORD *)(v44 + 8 * v45);
            goto LABEL_33;
          }
          v32 = *((_QWORD *)v33 + 11);
          if ( v32 == *((_QWORD *)v33 + 10) )
          {
            v31 = (_QWORD *)(v32 + 8LL * *((unsigned int *)v33 + 25));
            v34 = v31;
            goto LABEL_33;
          }
LABEL_27:
          v29 += 144;
          if ( v29 == v30 )
            goto LABEL_35;
        }
        v34 = &v31[*((unsigned int *)v33 + 25)];
        if ( v31 == v34 )
        {
LABEL_55:
          v31 = v34;
        }
        else
        {
          while ( v28 != *v31 )
          {
            if ( v34 == ++v31 )
              goto LABEL_55;
          }
        }
LABEL_33:
        if ( v31 == v34 )
          goto LABEL_27;
        v29 += 144;
        *v31 = -2;
        ++*((_DWORD *)v33 + 26);
      }
      while ( v29 != v30 );
LABEL_35:
      v20 = v28;
      v19 = v118;
      v26 = v110;
LABEL_36:
      v122 = 0;
      v125 = 4;
      v123 = (__int64 *)v127;
      v124 = (__int64 *)v127;
      v126 = 0;
      if ( (*(_BYTE *)(v19 - 1) & 0x40) != 0 )
      {
        v35 = *(__int64 ***)(v19 - 32);
        v26 = &v35[3 * (*(_DWORD *)(v19 - 4) & 0xFFFFFFF)];
      }
      else
      {
        v35 = (__int64 **)(v20 - 24LL * (*(_DWORD *)(v19 - 4) & 0xFFFFFFF));
      }
      v36 = sub_1992730(v35, v26, v1[5], v1[1]);
      if ( v36 != v26 )
      {
        v111 = v19;
        v37 = v1;
        v38 = v36;
        while ( 2 )
        {
          v40 = (__int64)*v38;
          v41 = v123;
          if ( v124 != v123 )
            goto LABEL_40;
          v42 = &v123[HIDWORD(v125)];
          if ( v123 != v42 )
          {
            v43 = 0;
            while ( v40 != *v41 )
            {
              if ( *v41 == -2 )
                v43 = v41;
              if ( v42 == ++v41 )
              {
                if ( !v43 )
                  goto LABEL_123;
                *v43 = v40;
                --v126;
                ++v122;
                goto LABEL_51;
              }
            }
            goto LABEL_41;
          }
LABEL_123:
          if ( HIDWORD(v125) < (unsigned int)v125 )
          {
            ++HIDWORD(v125);
            *v42 = v40;
            ++v122;
LABEL_51:
            sub_199B200(v37, v20, v40, (__int64)&v131);
          }
          else
          {
LABEL_40:
            sub_16CCBA0((__int64)&v122, (__int64)*v38);
            if ( v39 )
              goto LABEL_51;
          }
LABEL_41:
          v38 = sub_1992730(v38 + 3, v26, v37[5], v37[1]);
          if ( v38 == v26 )
          {
            v1 = v37;
            v19 = v111;
            break;
          }
          continue;
        }
      }
      if ( v124 == v123 )
      {
LABEL_14:
        v19 = *(_QWORD *)(v19 + 8);
        if ( v120 == v19 )
          goto LABEL_72;
      }
      else
      {
        _libc_free((unsigned __int64)v124);
        v19 = *(_QWORD *)(v19 + 8);
        if ( v120 == v19 )
          goto LABEL_72;
      }
    }
    v24 = &v23[*(unsigned int *)(v21 + 68)];
    if ( v23 == v24 )
    {
      v93 = *(_QWORD **)(*v1 + 48);
    }
    else
    {
      do
      {
        if ( v20 == *v23 )
          break;
        ++v23;
      }
      while ( v24 != v23 );
      v93 = v24;
    }
LABEL_61:
    while ( v93 != v23 )
    {
      if ( *v23 < 0xFFFFFFFFFFFFFFFELL )
        break;
      ++v23;
    }
    goto LABEL_21;
  }
LABEL_73:
  v48 = sub_157F280(**(_QWORD **)(v1[5] + 32));
  v50 = v49;
  v51 = v48;
  if ( v48 != v49 )
  {
    while ( !sub_1456C80(v1[1], *(_QWORD *)v51) )
    {
LABEL_85:
      v61 = *(_QWORD *)(v51 + 32);
      if ( !v61 )
        BUG();
      v51 = 0;
      if ( *(_BYTE *)(v61 - 8) == 77 )
        v51 = v61 - 24;
      if ( v50 == v51 )
        goto LABEL_89;
    }
    v52 = sub_13FCB50(v1[5]);
    v53 = *(_BYTE *)(v51 + 23) & 0x40;
    v54 = *(_DWORD *)(v51 + 20) & 0xFFFFFFF;
    if ( v54 )
    {
      v55 = 24LL * *(unsigned int *)(v51 + 56) + 8;
      v56 = 0;
      while ( 1 )
      {
        v57 = v51 - 24LL * v54;
        if ( v53 )
          v57 = *(_QWORD *)(v51 - 8);
        if ( v52 == *(_QWORD *)(v57 + v55) )
          break;
        ++v56;
        v55 += 8;
        if ( v54 == (_DWORD)v56 )
          goto LABEL_127;
      }
      v58 = 24 * v56;
      if ( v53 )
        goto LABEL_82;
    }
    else
    {
LABEL_127:
      v58 = 0x17FFFFFFE8LL;
      if ( v53 )
      {
LABEL_82:
        v59 = *(_QWORD *)(v51 - 8);
        goto LABEL_83;
      }
    }
    v59 = v51 - 24LL * v54;
LABEL_83:
    v60 = *(_QWORD *)(v59 + v58);
    if ( *(_BYTE *)(v60 + 16) > 0x17u )
      sub_199B200(v1, v51, v60, (__int64)&v131);
    goto LABEL_85;
  }
LABEL_89:
  v117 = *((_DWORD *)v1 + 8078);
  if ( !v117 )
    goto LABEL_147;
  v121 = 0;
  v62 = 0;
  v119 = 0;
  v63 = 0;
  do
  {
    v64 = (unsigned int *)(v62 + v1[4038]);
    v65 = v64[2];
    if ( (unsigned int)v65 > 1 && *(_DWORD *)&v131[3 * v62 + 28] == *(_DWORD *)&v131[3 * v62 + 32] )
    {
      v66 = *(_QWORD *)v64;
      v115 = 1;
      v67 = 24 * v65;
      v68 = *(_QWORD *)(*(_QWORD *)v64 + v67 - 24);
      if ( *(_BYTE *)(v68 + 16) == 77 )
      {
        v103 = sub_146F1B0(v1[1], v68);
        v66 = *(_QWORD *)v64;
        v115 = *(_QWORD *)(*(_QWORD *)v64 + 16LL) != v103;
        v67 = 24LL * v64[2];
      }
      v69 = v66 + 24;
      v70 = v66 + v67;
      if ( v66 + v67 != v66 + 24 )
      {
        v107 = v63;
        v109 = 0;
        v71 = 0;
        v108 = 0;
        v112 = 0;
        while ( 1 )
        {
          while ( sub_14560B0(*(_QWORD *)(v69 + 16)) )
          {
LABEL_100:
            v69 += 24;
            if ( v70 == v69 )
              goto LABEL_104;
          }
          v73 = *(_QWORD *)(v69 + 16);
          if ( *(_WORD *)(v73 + 24) )
          {
            if ( v73 == v71 )
            {
              ++v108;
            }
            else
            {
              ++v109;
              v71 = *(_QWORD *)(v69 + 16);
            }
            goto LABEL_100;
          }
          v69 += 24;
          ++v112;
          if ( v70 == v69 )
          {
LABEL_104:
            v63 = v107;
            if ( v109 - v108 + (v112 < 2) + v115 - 1 >= 0 )
              break;
            v74 = 48 * v107;
            if ( v119 != v121 )
            {
              v75 = v1[4038];
              v76 = v75 + v74;
              v77 = v75 + v62;
              if ( v75 + v74 != v75 + v62 )
              {
                v79 = *(unsigned int *)(v76 + 8);
                v116 = *(_DWORD *)(v77 + 8);
                v78 = v116;
                if ( v116 <= v79 )
                {
                  if ( v116 )
                    memmove(*(void **)v76, *(const void **)v77, 24LL * v116);
                }
                else
                {
                  if ( v116 > (unsigned __int64)*(unsigned int *)(v76 + 12) )
                  {
                    *(_DWORD *)(v76 + 8) = 0;
                    sub_16CD150(v76, (const void *)(v76 + 16), v116, 24, v75, v72);
                    v78 = *(unsigned int *)(v77 + 8);
                    v80 = 0;
                  }
                  else
                  {
                    v80 = 24 * v79;
                    if ( *(_DWORD *)(v76 + 8) )
                    {
                      v113 = 24 * v79;
                      memmove(*(void **)v76, *(const void **)v77, 24 * v79);
                      v78 = *(unsigned int *)(v77 + 8);
                      v80 = v113;
                    }
                  }
                  v81 = (const void *)(*(_QWORD *)v77 + v80);
                  if ( v81 != (const void *)(24 * v78 + *(_QWORD *)v77) )
                    memcpy((void *)(v80 + *(_QWORD *)v76), v81, 24 * v78 - v80);
                }
                *(_DWORD *)(v76 + 8) = v116;
              }
              *(_QWORD *)(v76 + 40) = *(_QWORD *)(v77 + 40);
            }
            v82 = 48 * v107 + v1[4038];
            v83 = (__int64 *)(*(_QWORD *)v82 + 24LL);
            v84 = *(_QWORD *)v82 + 24LL * *(unsigned int *)(v82 + 8);
            if ( (__int64 *)v84 == v83 )
            {
LABEL_138:
              v63 = (unsigned int)++v119;
              break;
            }
LABEL_121:
            while ( 2 )
            {
              v86 = *v83;
              v89 = 3LL * (*(_DWORD *)(*v83 + 20) & 0xFFFFFFF);
              if ( (*(_BYTE *)(*v83 + 23) & 0x40) != 0 )
              {
                v85 = *(_QWORD **)(v86 - 8);
                v86 = (__int64)&v85[v89];
              }
              else
              {
                v85 = (_QWORD *)(v86 - v89 * 8);
              }
              v87 = sub_1992F00(v85, v86, v83 + 1);
              v88 = (_QWORD *)v1[4089];
              if ( (_QWORD *)v1[4090] == v88 )
              {
                v90 = &v88[*((unsigned int *)v1 + 8183)];
                v91 = *((_DWORD *)v1 + 8183);
                if ( v88 == v90 )
                {
LABEL_139:
                  if ( v91 >= *((_DWORD *)v1 + 8182) )
                    goto LABEL_119;
                  *((_DWORD *)v1 + 8183) = v91 + 1;
                  *v90 = v87;
                  ++v1[4088];
                }
                else
                {
                  v92 = 0;
                  while ( v87 != (_QWORD *)*v88 )
                  {
                    if ( *v88 == -2 )
                      v92 = v88;
                    if ( v90 == ++v88 )
                    {
                      if ( !v92 )
                        goto LABEL_139;
                      v83 += 3;
                      *v92 = v87;
                      --*((_DWORD *)v1 + 8184);
                      ++v1[4088];
                      if ( (__int64 *)v84 != v83 )
                        goto LABEL_121;
                      goto LABEL_138;
                    }
                  }
                }
              }
              else
              {
LABEL_119:
                sub_16CCBA0((__int64)(v1 + 4088), (__int64)v87);
              }
              v83 += 3;
              if ( (__int64 *)v84 == v83 )
                goto LABEL_138;
              continue;
            }
          }
        }
      }
    }
    ++v121;
    v62 += 48;
  }
  while ( v121 != v117 );
  v94 = *((unsigned int *)v1 + 8078);
  if ( v63 < v94 )
  {
    v95 = (unsigned __int64 *)(v1[4038] + 48 * v94);
    v96 = v1[4038] + 48 * v63;
    while ( (unsigned __int64 *)v96 != v95 )
    {
      v95 -= 6;
      if ( (unsigned __int64 *)*v95 != v95 + 2 )
        _libc_free(*v95);
    }
    goto LABEL_146;
  }
  if ( v63 > v94 )
  {
    if ( v63 > *((unsigned int *)v1 + 8079) )
    {
      sub_1995CB0((__int64)(v1 + 4038), v63);
      v94 = *((unsigned int *)v1 + 8078);
    }
    v101 = v1[4038] + 48 * v94;
    v102 = v1[4038] + 48 * v63;
    if ( v101 == v102 )
    {
LABEL_146:
      *((_DWORD *)v1 + 8078) = v119;
    }
    else
    {
      do
      {
        if ( v101 )
        {
          *(_DWORD *)(v101 + 8) = 0;
          *(_QWORD *)v101 = v101 + 16;
          *(_DWORD *)(v101 + 12) = 1;
          *(_OWORD *)(v101 + 16) = 0;
          *(_OWORD *)(v101 + 32) = 0;
        }
        v101 += 48;
      }
      while ( v102 != v101 );
      *((_DWORD *)v1 + 8078) = v119;
    }
  }
LABEL_147:
  if ( v128 != v130 )
    _libc_free((unsigned __int64)v128);
  v97 = v131;
  v98 = (unsigned __int64)&v131[144 * (unsigned int)v132];
  if ( v131 != (_BYTE *)v98 )
  {
    do
    {
      v98 -= 144LL;
      v99 = *(_QWORD *)(v98 + 88);
      if ( v99 != *(_QWORD *)(v98 + 80) )
        _libc_free(v99);
      v100 = *(_QWORD *)(v98 + 16);
      if ( v100 != *(_QWORD *)(v98 + 8) )
        _libc_free(v100);
    }
    while ( v97 != (_BYTE *)v98 );
    v98 = (unsigned __int64)v131;
  }
  if ( (_BYTE *)v98 != v133 )
    _libc_free(v98);
}
