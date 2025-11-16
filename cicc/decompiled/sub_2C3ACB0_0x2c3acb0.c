// Function: sub_2C3ACB0
// Address: 0x2c3acb0
//
_QWORD *__fastcall sub_2C3ACB0(__int64 a1)
{
  __int64 v1; // rdx
  __int64 *v2; // rax
  __int64 *v3; // rdx
  __int64 v4; // rax
  __int64 v5; // rbx
  __int64 v6; // rax
  _QWORD *v7; // r12
  _QWORD *v8; // r13
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 v11; // r8
  __int64 i; // rsi
  __int64 *v13; // rax
  __int64 *v14; // r10
  __int64 v15; // rdi
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rbx
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // r13
  __int64 v25; // r12
  __int64 *v26; // r14
  __int64 *v27; // r12
  __int64 v28; // rcx
  unsigned __int8 v29; // al
  __int64 v30; // r13
  const char *v31; // r15
  _BYTE *v32; // rax
  __int64 v33; // rdx
  _BYTE *v34; // rcx
  __int64 v35; // rsi
  __int64 v36; // rdx
  _BYTE *v37; // rdx
  char v38; // si
  __int64 v39; // rdi
  __int64 v40; // rax
  __int64 v41; // rbx
  __int64 v42; // r13
  int v43; // r14d
  const char **v44; // r10
  unsigned int v45; // ecx
  const char **v46; // rax
  const char *v47; // r9
  __int64 v48; // r12
  const char *v49; // rdx
  int v50; // esi
  int v51; // ecx
  __int64 v52; // rax
  __int64 v53; // r15
  char v54; // al
  __int64 v55; // rcx
  char v56; // al
  __int64 v57; // rax
  __int64 v58; // r9
  _QWORD *v59; // rcx
  __int64 v60; // r11
  __int64 v61; // rsi
  __int64 *v62; // rax
  __int64 v63; // r9
  __int64 v64; // r10
  __int64 v65; // rcx
  __int64 v66; // r8
  int v67; // edx
  char v68; // al
  unsigned __int8 *v69; // rax
  __int64 v70; // rsi
  __int64 v71; // rax
  _QWORD *result; // rax
  __int64 v73; // r15
  __int64 v74; // rax
  __int64 v75; // rbx
  __int64 v76; // rax
  __int64 v77; // rbx
  __int64 v78; // r13
  _QWORD *v79; // r14
  __int64 v80; // r12
  __int64 v81; // rax
  __int64 v82; // r12
  __int64 v83; // r15
  _QWORD *v84; // r12
  __int64 v85; // r12
  __int64 v86; // rax
  __int64 v87; // rax
  __int64 v88; // r15
  __int64 v89; // rdx
  __int64 v90; // rdx
  __int64 v91; // rsi
  __int64 v92; // r14
  char v93; // al
  _QWORD *v94; // r14
  __int64 v95; // rax
  _QWORD *v96; // rsi
  __int64 v97; // rax
  _QWORD *v98; // r13
  __int64 v99; // rax
  _QWORD *v100; // r12
  char v101; // al
  char v102; // al
  __int64 v103; // rdi
  _QWORD *v104; // rdx
  __int64 (__fastcall *v105)(__int64, __int64); // rax
  char v106; // al
  bool v107; // [rsp+Fh] [rbp-101h]
  __int64 v108; // [rsp+18h] [rbp-F8h]
  __int64 v109; // [rsp+18h] [rbp-F8h]
  __int64 v110; // [rsp+20h] [rbp-F0h]
  __int64 v111; // [rsp+20h] [rbp-F0h]
  unsigned __int8 *v112; // [rsp+28h] [rbp-E8h]
  __int64 v113; // [rsp+28h] [rbp-E8h]
  _QWORD *v114; // [rsp+30h] [rbp-E0h]
  __int64 v115; // [rsp+30h] [rbp-E0h]
  __int64 v116; // [rsp+30h] [rbp-E0h]
  _QWORD *v117; // [rsp+30h] [rbp-E0h]
  __int64 v118; // [rsp+38h] [rbp-D8h]
  __int64 v119; // [rsp+38h] [rbp-D8h]
  __int64 v120; // [rsp+40h] [rbp-D0h]
  _QWORD *v121; // [rsp+40h] [rbp-D0h]
  __int64 v122; // [rsp+40h] [rbp-D0h]
  __int64 v123; // [rsp+40h] [rbp-D0h]
  __int64 v124; // [rsp+40h] [rbp-D0h]
  _QWORD *v125; // [rsp+40h] [rbp-D0h]
  __int64 v126; // [rsp+40h] [rbp-D0h]
  __int64 v127; // [rsp+40h] [rbp-D0h]
  __int64 v129; // [rsp+58h] [rbp-B8h] BYREF
  _QWORD v130[2]; // [rsp+60h] [rbp-B0h] BYREF
  const char *v131; // [rsp+70h] [rbp-A0h] BYREF
  const char *v132; // [rsp+78h] [rbp-98h]
  char v133; // [rsp+90h] [rbp-80h]
  char v134; // [rsp+91h] [rbp-7Fh]
  __int64 *v135; // [rsp+A0h] [rbp-70h] BYREF
  __int64 v136; // [rsp+A8h] [rbp-68h]
  __int64 v137; // [rsp+B0h] [rbp-60h] BYREF
  unsigned int v138; // [rsp+B8h] [rbp-58h]

  v1 = sub_2AAFF80(a1);
  v2 = *(__int64 **)(v1 + 112);
  v3 = &v2[*(unsigned int *)(v1 + 120)];
  if ( v2 == v3 )
    goto LABEL_4;
  while ( *(_BYTE *)(*v2 - 32) != 15 )
  {
    if ( v3 == ++v2 )
      goto LABEL_4;
  }
  v73 = *v2;
  v115 = *v2;
  v74 = sub_2BF3F10((_QWORD *)a1);
  v75 = sub_2BF04D0(v74);
  v76 = sub_2BF05A0(v75);
  v77 = *(_QWORD *)(v75 + 120);
  v78 = v76;
  v119 = v73 + 56;
  if ( v77 == v76 )
  {
LABEL_4:
    if ( !LOBYTE(qword_500D260[17]) )
      goto LABEL_5;
  }
  else
  {
    do
    {
      if ( !v77 )
        goto LABEL_209;
      if ( *(_BYTE *)(v77 - 16) == 33 && sub_2C1B260(v77 - 24) )
      {
        v79 = *(_QWORD **)(v77 + 88);
        v80 = 8LL * *(unsigned int *)(v77 + 96);
        v125 = &v79[(unsigned __int64)v80 / 8];
        v81 = v80 >> 3;
        v82 = v80 >> 5;
        if ( !v82 )
          goto LABEL_138;
        v83 = v77 + 72;
        v84 = &v79[4 * v82];
        do
        {
          if ( !(*(unsigned __int8 (__fastcall **)(_QWORD, __int64))(*(_QWORD *)*v79 + 16LL))(*v79, v77 + 72) )
            goto LABEL_117;
          if ( !(*(unsigned __int8 (__fastcall **)(_QWORD, __int64))(*(_QWORD *)v79[1] + 16LL))(v79[1], v77 + 72) )
          {
            if ( v125 == v79 + 1 )
              goto LABEL_118;
            goto LABEL_123;
          }
          if ( !(*(unsigned __int8 (__fastcall **)(_QWORD, __int64))(*(_QWORD *)v79[2] + 16LL))(v79[2], v77 + 72) )
          {
            v79 += 2;
            goto LABEL_117;
          }
          if ( !(*(unsigned __int8 (__fastcall **)(_QWORD, __int64))(*(_QWORD *)v79[3] + 16LL))(v79[3], v77 + 72) )
          {
            v79 += 3;
            goto LABEL_117;
          }
          v79 += 4;
        }
        while ( v84 != v79 );
        v81 = v125 - v79;
LABEL_138:
        if ( v81 != 2 )
        {
          if ( v81 != 3 )
          {
            if ( v81 == 1 )
            {
              v83 = v77 + 72;
              v93 = (*(__int64 (__fastcall **)(_QWORD, __int64))(*(_QWORD *)*v79 + 16LL))(*v79, v77 + 72);
              goto LABEL_187;
            }
            goto LABEL_118;
          }
          v83 = v77 + 72;
          if ( (*(unsigned __int8 (__fastcall **)(_QWORD, __int64))(*(_QWORD *)*v79 + 16LL))(*v79, v77 + 72) )
          {
            v100 = v79 + 1;
            v101 = (*(__int64 (__fastcall **)(_QWORD, __int64))(*(_QWORD *)v79[1] + 16LL))(v79[1], v83);
            goto LABEL_185;
          }
LABEL_117:
          if ( v125 == v79 )
            goto LABEL_118;
LABEL_123:
          v85 = v115 + 56;
          goto LABEL_124;
        }
        v83 = v77 + 72;
        v100 = v79;
        v101 = (*(__int64 (__fastcall **)(_QWORD, __int64))(*(_QWORD *)*v79 + 16LL))(*v79, v77 + 72);
LABEL_185:
        if ( !v101 )
        {
          v79 = v100;
          goto LABEL_117;
        }
        v79 = v100 + 1;
        v93 = (*(__int64 (__fastcall **)(_QWORD, __int64))(*(_QWORD *)v100[1] + 16LL))(v100[1], v83);
LABEL_187:
        if ( !v93 )
          goto LABEL_117;
LABEL_118:
        v85 = v119;
        if ( (unsigned __int8)sub_2C46C30(v119) )
        {
          v83 = v77 + 72;
LABEL_124:
          sub_2BF1250(v85, v83);
          sub_2C19E60((__int64 *)(v115 - 40));
          goto LABEL_4;
        }
      }
      v77 = *(_QWORD *)(v77 + 8);
    }
    while ( v77 != v78 );
    if ( !LOBYTE(qword_500D260[17]) )
      goto LABEL_5;
  }
  sub_2C4B640(a1);
LABEL_5:
  v4 = sub_2BF3F10((_QWORD *)a1);
  v5 = sub_2BF04D0(v4);
  v6 = sub_2BF05A0(v5);
  v7 = *(_QWORD **)(v5 + 120);
  v8 = (_QWORD *)v6;
  if ( v7 != (_QWORD *)v6 )
  {
    while ( v7 )
    {
      if ( *((_BYTE *)v7 - 16) != 33 || v7[17] )
      {
        v7 = (_QWORD *)v7[1];
        if ( v7 == v8 )
          goto LABEL_20;
      }
      else
      {
        v9 = v7[16];
        v10 = (__int64)(v7 + 9);
        v11 = *(_QWORD *)(v9 + 48);
        for ( i = v11 + 8LL * *(unsigned int *)(v9 + 56); v11 != i; i -= 8 )
        {
          v13 = *(__int64 **)(v10 + 16);
          v14 = &v13[*(unsigned int *)(v10 + 24)];
          if ( v13 == v14 )
          {
LABEL_17:
            v10 = 0;
          }
          else
          {
            while ( 1 )
            {
              v15 = *v13;
              if ( *v13 )
                break;
LABEL_16:
              if ( v14 == ++v13 )
                goto LABEL_17;
            }
            switch ( *(_BYTE *)(v15 - 32) )
            {
              case 0:
              case 3:
              case 5:
              case 0x13:
              case 0x14:
              case 0x15:
              case 0x16:
              case 0x1A:
                goto LABEL_16;
              case 1:
              case 2:
              case 4:
              case 6:
              case 7:
              case 8:
              case 9:
              case 0xA:
              case 0xB:
              case 0xC:
              case 0xD:
              case 0xE:
              case 0xF:
              case 0x10:
              case 0x11:
              case 0x12:
              case 0x17:
              case 0x18:
              case 0x19:
              case 0x1B:
              case 0x1C:
              case 0x1D:
              case 0x1E:
              case 0x1F:
              case 0x20:
              case 0x21:
              case 0x22:
              case 0x23:
              case 0x24:
                if ( *(_QWORD *)(i - 8) != *(_QWORD *)(v15 + 96) )
                  goto LABEL_16;
                v10 = v15 + 56;
                break;
              default:
                goto LABEL_209;
            }
          }
        }
        sub_2BF1250(v10, (__int64)(v7 + 9));
        v7 = (_QWORD *)v7[1];
        if ( v7 == v8 )
          goto LABEL_20;
      }
    }
LABEL_209:
    BUG();
  }
LABEL_20:
  if ( LOBYTE(qword_500D260[17]) )
    sub_2C4B640(a1);
  v16 = sub_2AAFF80(a1);
  if ( !*(_DWORD *)(v16 + 56) )
LABEL_210:
    BUG();
  sub_2C36780((__int64 *)a1, *(_QWORD **)(*(_QWORD *)(**(_QWORD **)(v16 + 48) + 40LL) + 8LL));
  if ( LOBYTE(qword_500D260[17]) )
    sub_2C4B640(a1);
  sub_2C37F10((__int64 *)a1);
  if ( LOBYTE(qword_500D260[17]) )
    sub_2C4B640(a1);
  v17 = sub_2BF3F10((_QWORD *)a1);
  v107 = 0;
  v18 = sub_2BF04D0(v17);
  if ( *(_DWORD *)(a1 + 88) == 1 )
  {
    v99 = *(_QWORD *)(a1 + 80);
    if ( !*(_BYTE *)(v99 + 4) )
      v107 = *(_DWORD *)v99 == 1;
  }
  v130[0] = v18;
  v130[1] = sub_2BF05A0(v18);
  v19 = sub_2BF05A0(v18);
  v24 = *(_QWORD *)(v18 + 120);
  v118 = v19;
  if ( v24 != v19 )
  {
    while ( 1 )
    {
      if ( !v24 )
        goto LABEL_209;
      if ( (unsigned int)*(unsigned __int8 *)(v24 - 16) - 33 <= 1 )
        break;
LABEL_51:
      v24 = *(_QWORD *)(v24 + 8);
      if ( v24 == v118 )
        goto LABEL_52;
    }
    v25 = v24 + 72;
    sub_2C3A0F0(&v135, v24 + 72, v20, v21, v22, v23);
    v26 = v135;
    if ( v135 != &v135[(unsigned int)v136] )
    {
      v110 = v24;
      v108 = v24 + 72;
      v27 = &v135[(unsigned int)v136];
      while ( 2 )
      {
        v28 = *(v27 - 1);
        v29 = *(_BYTE *)(v28 - 32);
        v20 = v29;
        switch ( v29 )
        {
          case 0u:
          case 3u:
          case 5u:
          case 0x13u:
          case 0x14u:
          case 0x15u:
          case 0x16u:
          case 0x1Au:
            goto LABEL_37;
          case 1u:
          case 2u:
          case 4u:
          case 6u:
          case 7u:
          case 8u:
          case 9u:
          case 0xAu:
          case 0xBu:
          case 0xCu:
          case 0xDu:
          case 0xEu:
          case 0xFu:
          case 0x10u:
          case 0x11u:
          case 0x12u:
          case 0x17u:
          case 0x18u:
          case 0x19u:
          case 0x1Bu:
          case 0x1Cu:
          case 0x1Du:
          case 0x1Eu:
          case 0x1Fu:
          case 0x20u:
          case 0x21u:
          case 0x22u:
          case 0x23u:
          case 0x24u:
            v30 = v28 - 40;
            if ( v29 == 9 )
            {
              v52 = v28 - 40;
            }
            else
            {
              if ( v29 != 23 )
                goto LABEL_37;
              v52 = 0;
            }
            v23 = *(unsigned int *)(v28 + 80);
            if ( (_DWORD)v23 && *(_QWORD *)(v28 + 96) && (!v52 || !*(_BYTE *)(v52 + 160) && !*(_BYTE *)(v52 + 161)) )
            {
              v53 = v28 + 56;
              v120 = *(v27 - 1);
              v54 = sub_2AAA120(v28 + 56);
              v55 = v120;
              if ( v54 || (v56 = sub_2C46C30(v53), v55 = v120, v56) )
              {
                v114 = *(_QWORD **)(v55 + 8);
                v112 = *(unsigned __int8 **)(v55 + 96);
                v121 = &v114[*(unsigned int *)(v55 + 16)];
                v57 = sub_22077B0(0xA8u);
                v59 = v121;
                if ( v57 )
                {
                  v122 = v57;
                  sub_2ABDBC0(v57, 9, v114, v59, v112, v58);
                  *(_WORD *)(v122 + 160) = 1;
                  *(_QWORD *)v122 = &unk_4A237B0;
                  *(_QWORD *)(v122 + 40) = &unk_4A237F8;
                  *(_QWORD *)(v122 + 96) = &unk_4A23830;
                  sub_2C19DE0((_QWORD *)v122, v30);
                  v60 = v122 + 96;
                }
                else
                {
                  sub_2C19DE0(0, v30);
                  v60 = 0;
                }
                sub_2BF1250(v53, v60);
              }
            }
LABEL_37:
            if ( v26 != --v27 )
              continue;
            v24 = v110;
            v25 = v108;
            break;
          default:
            goto LABEL_209;
        }
        break;
      }
    }
    v31 = (const char *)(v24 - 24);
    if ( *(_BYTE *)(v24 - 16) == 34 )
    {
      v32 = *(_BYTE **)(a1 + 80);
      v33 = 8LL * *(unsigned int *)(a1 + 88);
      v34 = &v32[v33];
      v35 = v33 >> 3;
      v36 = v33 >> 5;
      if ( v36 )
      {
        v37 = &v32[32 * v36];
        while ( 1 )
        {
          if ( v32[4] )
          {
            v38 = v32 != v34;
            goto LABEL_48;
          }
          if ( v32[12] )
          {
            v38 = v34 != v32 + 8;
            goto LABEL_48;
          }
          if ( v32[20] )
          {
            v38 = v34 != v32 + 16;
            goto LABEL_48;
          }
          if ( v32[28] )
            break;
          v32 += 32;
          if ( v37 == v32 )
          {
            v35 = (v34 - v32) >> 3;
            goto LABEL_154;
          }
        }
        v38 = v34 != v32 + 24;
        goto LABEL_48;
      }
LABEL_154:
      if ( v35 != 2 )
      {
        if ( v35 != 3 )
        {
          if ( v35 != 1 )
          {
            v38 = 0;
            goto LABEL_48;
          }
LABEL_163:
          v38 = 0;
          if ( !v32[4] )
            goto LABEL_48;
          goto LABEL_164;
        }
        if ( v32[4] )
        {
LABEL_164:
          v38 = v34 != v32;
LABEL_48:
          if ( (unsigned __int8)sub_2C1B680(v24 - 24, v38) )
          {
            v86 = sub_D95540(*(_QWORD *)(*(_QWORD *)(v24 + 128) + 32LL));
            v87 = sub_AD64C0(v86, 0, 0);
            v88 = sub_2AC42A0(a1, v87);
            v89 = *(_QWORD *)(*(_QWORD *)(v24 + 24) + 8LL);
            v131 = *(const char **)(v24 + 64);
            if ( v131 )
            {
              v126 = v89;
              sub_2C25AB0((__int64 *)&v131);
              v89 = v126;
            }
            v127 = sub_2C28C90((_QWORD *)a1, 1, 13, 0, 0, v88, v89, (__int64 *)&v131, v130);
            sub_9C6650(&v131);
            v134 = 1;
            v90 = v127;
            v131 = "next.gep";
            v133 = 3;
            v129 = *(_QWORD *)(v24 + 64);
            if ( v129 )
            {
              sub_2C25AB0(&v129);
              v90 = v127;
            }
            if ( v90 )
              v90 += 96;
            v91 = 0;
            if ( *(_DWORD *)(v24 + 32) )
              v91 = **(_QWORD **)(v24 + 24);
            v92 = sub_2C286A0(v130, v91, v90, &v129, (void **)&v131);
            if ( v92 )
              v92 += 96;
            sub_9C6650(&v129);
            sub_2BF1250(v25, v92);
          }
          goto LABEL_49;
        }
        v32 += 8;
      }
      if ( !v32[4] )
      {
        v32 += 8;
        goto LABEL_163;
      }
      goto LABEL_164;
    }
    if ( v107 )
    {
LABEL_77:
      v61 = *(_QWORD *)(v24 + 128);
      v131 = *(const char **)(v24 + 64);
      if ( v131 )
        sub_2C25AB0((__int64 *)&v131);
      v62 = *(__int64 **)(v24 + 24);
      v63 = 0;
      v64 = v62[1];
      if ( *(_DWORD *)(v24 + 32) )
        v63 = *v62;
      v65 = *(_QWORD *)(v61 + 40);
      v66 = *(_QWORD *)(v24 + 136);
      v67 = 31;
      if ( v65 )
      {
        v109 = *(_QWORD *)(v24 + 136);
        v111 = v63;
        v113 = v62[1];
        v123 = *(_QWORD *)(v61 + 40);
        v68 = sub_920620(v123);
        v65 = v123;
        v64 = v113;
        v63 = v111;
        if ( !v68 )
          v65 = 0;
        v69 = *(unsigned __int8 **)(v61 + 40);
        v66 = v109;
        v67 = 31;
        if ( v69 )
          v67 = *v69 - 29;
      }
      v124 = sub_2C28C90((_QWORD *)a1, *(_DWORD *)(v61 + 24), v67, v65, v66, v63, v64, (__int64 *)&v131, v130);
      sub_9C6650(&v131);
      v70 = v124;
      if ( v107 )
      {
        if ( v124 )
          v70 = v124 + 96;
        sub_2BF1250(v25, v70);
      }
      else
      {
        if ( v124 )
          v70 = v124 + 96;
        v131 = v31;
        sub_2BF1090(v25, v70, (unsigned __int8 (__fastcall *)(__int64, __int64, _QWORD))sub_2C250C0, (__int64)&v131);
      }
      goto LABEL_49;
    }
    v94 = *(_QWORD **)(v24 + 88);
    v95 = 8LL * *(unsigned int *)(v24 + 96);
    v96 = &v94[(unsigned __int64)v95 / 8];
    v21 = v95 >> 3;
    v97 = v95 >> 5;
    if ( v97 )
    {
      v116 = v24;
      v98 = &v94[4 * v97];
      while ( 1 )
      {
        if ( (*(unsigned __int8 (__fastcall **)(_QWORD, __int64, __int64, __int64))(*(_QWORD *)*v94 + 16LL))(
               *v94,
               v25,
               v20,
               v21) )
        {
          v24 = v116;
          goto LABEL_150;
        }
        if ( (*(unsigned __int8 (__fastcall **)(_QWORD, __int64))(*(_QWORD *)v94[1] + 16LL))(v94[1], v25) )
        {
          v24 = v116;
          ++v94;
          goto LABEL_150;
        }
        if ( (*(unsigned __int8 (__fastcall **)(_QWORD, __int64))(*(_QWORD *)v94[2] + 16LL))(v94[2], v25) )
        {
          v24 = v116;
          v94 += 2;
          goto LABEL_150;
        }
        if ( (*(unsigned __int8 (__fastcall **)(_QWORD, __int64))(*(_QWORD *)v94[3] + 16LL))(v94[3], v25) )
          break;
        v94 += 4;
        if ( v98 == v94 )
        {
          v24 = v116;
          v21 = v96 - v94;
          goto LABEL_191;
        }
      }
      v24 = v116;
      v94 += 3;
LABEL_150:
      if ( v96 != v94 )
        goto LABEL_77;
LABEL_49:
      if ( v135 != &v137 )
        _libc_free((unsigned __int64)v135);
      goto LABEL_51;
    }
LABEL_191:
    switch ( v21 )
    {
      case 2LL:
        v103 = *v94;
        v104 = v94;
        v105 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)*v94 + 16LL);
        break;
      case 3LL:
        if ( (*(unsigned __int8 (__fastcall **)(_QWORD, __int64))(*(_QWORD *)*v94 + 16LL))(*v94, v25) )
          goto LABEL_150;
        v103 = v94[1];
        v104 = v94 + 1;
        v105 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v103 + 16LL);
        break;
      case 1LL:
        v102 = (*(__int64 (__fastcall **)(_QWORD, __int64))(*(_QWORD *)*v94 + 16LL))(*v94, v25);
        goto LABEL_195;
      default:
        goto LABEL_49;
    }
    v117 = v104;
    v106 = v105(v103, v25);
    v20 = (__int64)v117;
    if ( v106 )
    {
      v94 = v117;
      goto LABEL_150;
    }
    v94 = v117 + 1;
    v102 = (*(__int64 (__fastcall **)(_QWORD, __int64))(*(_QWORD *)v117[1] + 16LL))(v117[1], v25);
LABEL_195:
    if ( !v102 )
      goto LABEL_49;
    goto LABEL_150;
  }
LABEL_52:
  if ( LOBYTE(qword_500D260[17]) )
    sub_2C4B640(a1);
  v135 = 0;
  v136 = 0;
  v39 = *(_QWORD *)a1;
  v137 = 0;
  v138 = 0;
  v40 = sub_2BF04D0(v39);
  v41 = *(_QWORD *)(v40 + 120);
  v42 = v40 + 112;
  while ( v42 != v41 )
  {
    v48 = v41;
    v41 = *(_QWORD *)(v41 + 8);
    if ( *(_BYTE *)(v48 - 16) == 2 )
    {
      v49 = *(const char **)(v48 + 128);
      v50 = v138;
      v132 = (const char *)(v48 + 72);
      v131 = v49;
      if ( !v138 )
      {
        v135 = (__int64 *)((char *)v135 + 1);
        v130[0] = 0;
        goto LABEL_62;
      }
      v43 = 1;
      v44 = 0;
      v45 = (v138 - 1) & (((unsigned int)v49 >> 9) ^ ((unsigned int)v49 >> 4));
      v46 = (const char **)(v136 + 16LL * v45);
      v47 = *v46;
      if ( v49 != *v46 )
      {
        while ( v47 != (const char *)-4096LL )
        {
          if ( !v44 && v47 == (const char *)-8192LL )
            v44 = v46;
          v45 = (v138 - 1) & (v43 + v45);
          v46 = (const char **)(v136 + 16LL * v45);
          v47 = *v46;
          if ( v49 == *v46 )
            goto LABEL_57;
          ++v43;
        }
        if ( !v44 )
          v44 = v46;
        v135 = (__int64 *)((char *)v135 + 1);
        v51 = v137 + 1;
        v130[0] = v44;
        if ( 4 * ((int)v137 + 1) >= 3 * v138 )
        {
LABEL_62:
          v50 = 2 * v138;
        }
        else if ( v138 - HIDWORD(v137) - v51 > v138 >> 3 )
        {
          goto LABEL_180;
        }
        sub_2C2E1E0((__int64)&v135, v50);
        sub_2C2BBF0((__int64)&v135, (__int64 *)&v131, v130);
        v49 = v131;
        v44 = (const char **)v130[0];
        v51 = v137 + 1;
LABEL_180:
        LODWORD(v137) = v51;
        if ( *v44 != (const char *)-4096LL )
          --HIDWORD(v137);
        *v44 = v49;
        v44[1] = v132;
        continue;
      }
LABEL_57:
      sub_2BF1250(v48 + 72, (__int64)v46[1]);
      sub_2C19E60((__int64 *)(v48 - 24));
    }
  }
  sub_C7D6A0(v136, 16LL * v138, 8);
  if ( LOBYTE(qword_500D260[17]) )
    sub_2C4B640(a1);
  v71 = sub_2AAFF80(a1);
  if ( !*(_DWORD *)(v71 + 56) )
    goto LABEL_210;
  sub_2C36780((__int64 *)a1, *(_QWORD **)(*(_QWORD *)(**(_QWORD **)(v71 + 48) + 40LL) + 8LL));
  if ( LOBYTE(qword_500D260[17]) )
    sub_2C4B640(a1);
  sub_2C37F10((__int64 *)a1);
  if ( LOBYTE(qword_500D260[17]) )
    sub_2C4B640(a1);
  sub_2C39EE0((__int64 *)a1);
  if ( LOBYTE(qword_500D260[17]) )
    sub_2C4B640(a1);
  sub_2C34510((__int64 *)a1);
  if ( LOBYTE(qword_500D260[17]) )
    sub_2C4B640(a1);
  sub_2C2F9F0((_QWORD *)a1);
  result = qword_500D260;
  if ( LOBYTE(qword_500D260[17]) )
    return (_QWORD *)sub_2C4B640(a1);
  return result;
}
