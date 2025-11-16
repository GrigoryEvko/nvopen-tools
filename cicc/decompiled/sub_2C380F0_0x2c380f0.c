// Function: sub_2C380F0
// Address: 0x2c380f0
//
void __fastcall sub_2C380F0(
        __int64 *a1,
        __int64 (__fastcall *a2)(__int64, __int64),
        __int64 a3,
        __int64 a4,
        __int64 *a5)
{
  __int64 v5; // rax
  __int64 v6; // rbx
  unsigned __int64 v7; // rax
  bool v8; // zf
  unsigned __int64 v9; // rax
  __int64 v10; // r14
  __int64 v11; // rax
  __int64 v12; // rbx
  __int64 v13; // r12
  __int64 v14; // rax
  __int64 v15; // r13
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // r15
  __int64 v19; // rbx
  __int64 v20; // rcx
  __int64 v21; // rdx
  __int64 v22; // r9
  __int64 v23; // rsi
  __int64 v24; // rsi
  __int64 v25; // rdx
  __int64 v26; // rdx
  __int64 v27; // rdx
  __int64 v28; // rdx
  __int64 v29; // rax
  unsigned __int64 v30; // rdx
  unsigned __int64 v31; // rsi
  __int64 v32; // r15
  unsigned __int8 *v33; // r13
  int v34; // eax
  __int64 v35; // rbx
  __int64 v36; // r9
  __int64 v37; // rdx
  __int64 v38; // rax
  __int64 v39; // r8
  __int64 v40; // r9
  __int64 *v41; // rax
  __int64 v42; // rbx
  __int64 v43; // r12
  __int64 v44; // rax
  __int64 *v45; // r8
  __int64 *v46; // r9
  const void *v47; // rcx
  __int64 v48; // rdx
  __int64 v49; // rsi
  __int64 v50; // rbx
  __int64 *v51; // r12
  __int64 v52; // rax
  __int64 v53; // r15
  __int64 v54; // r13
  __int64 v55; // rcx
  __int64 v56; // rax
  __int64 v57; // rdx
  _QWORD *v58; // r12
  _QWORD *v59; // rbx
  __int64 v60; // rax
  __int64 v61; // r9
  char *v62; // rax
  int v63; // eax
  __int64 *v64; // r12
  __int64 v65; // rbx
  unsigned __int64 v66; // rax
  __int16 v67; // ax
  __int64 *v68; // rbx
  __int64 *v69; // r12
  __int64 v70; // rax
  __int64 v71; // r9
  char v72; // al
  __int64 v73; // rdx
  __int64 v74; // rbx
  int v75; // r12d
  __int64 v76; // rax
  __int64 v77; // r8
  __int64 v78; // rsi
  __int64 *v79; // r12
  __int64 v80; // rax
  __int64 v81; // r9
  int v82; // eax
  unsigned __int8 v83; // al
  int v84; // eax
  __int64 v86; // [rsp+10h] [rbp-200h]
  __int64 v87; // [rsp+18h] [rbp-1F8h]
  __int64 v89; // [rsp+30h] [rbp-1E0h]
  __int64 v90; // [rsp+30h] [rbp-1E0h]
  const void *v91; // [rsp+38h] [rbp-1D8h]
  const void *v92; // [rsp+38h] [rbp-1D8h]
  __int64 v95; // [rsp+50h] [rbp-1C0h]
  __int64 v96; // [rsp+58h] [rbp-1B8h]
  __int64 v97; // [rsp+60h] [rbp-1B0h]
  __int64 v98; // [rsp+68h] [rbp-1A8h]
  unsigned __int8 *v99; // [rsp+68h] [rbp-1A8h]
  __int64 v100; // [rsp+68h] [rbp-1A8h]
  unsigned __int64 v101; // [rsp+80h] [rbp-190h]
  __int64 v102; // [rsp+88h] [rbp-188h]
  int v103; // [rsp+88h] [rbp-188h]
  __int64 v104; // [rsp+88h] [rbp-188h]
  unsigned __int64 v105; // [rsp+90h] [rbp-180h]
  __int64 *v106; // [rsp+98h] [rbp-178h]
  _BYTE *v107; // [rsp+A8h] [rbp-168h] BYREF
  _BYTE *v108; // [rsp+B0h] [rbp-160h] BYREF
  _BYTE *v109; // [rsp+B8h] [rbp-158h] BYREF
  _BYTE *v110; // [rsp+C0h] [rbp-150h] BYREF
  char v111; // [rsp+C9h] [rbp-147h]
  unsigned __int64 v112; // [rsp+D0h] [rbp-140h]
  char v113; // [rsp+D9h] [rbp-137h]
  _BYTE *v114; // [rsp+E0h] [rbp-130h] BYREF
  __int64 v115; // [rsp+E8h] [rbp-128h]
  _BYTE v116[64]; // [rsp+F0h] [rbp-120h] BYREF
  _BYTE *v117; // [rsp+130h] [rbp-E0h] BYREF
  __int64 v118; // [rsp+138h] [rbp-D8h]
  char v119; // [rsp+140h] [rbp-D0h] BYREF
  _QWORD v120[6]; // [rsp+180h] [rbp-90h] BYREF
  __int64 v121; // [rsp+1B0h] [rbp-60h]

  v5 = sub_2BF3F10((_QWORD *)*a1);
  v114 = v116;
  v115 = 0x800000000LL;
  sub_2C363F0((__int64)&v114, v5);
  v112 = (unsigned __int64)v114;
  v110 = &v114[8 * (unsigned int)v115];
  v111 = 1;
  v113 = 1;
  sub_2C26110((__int64)&v117, (__int64 *)&v110);
  sub_2C25F40((__int64)v120, (__int64 *)&v117);
  v87 = v120[2];
  v95 = v120[0];
  v86 = v121;
  if ( v120[0] != v121 )
  {
    while ( 1 )
    {
      v6 = *(_QWORD *)(v95 - 8);
      if ( !*(_QWORD *)(v6 + 48) )
        goto LABEL_90;
      v7 = sub_2BF0A50(*(_QWORD *)(v95 - 8));
      v8 = v7 == 0;
      v9 = v7 + 24;
      v10 = *(_QWORD *)(v6 + 120);
      if ( v8 )
        v9 = v6 + 112;
      v101 = v9;
      if ( v9 != *(_QWORD *)(v6 + 120) )
        break;
LABEL_85:
      v95 -= 8;
      if ( v87 != v95 )
      {
        v57 = v95;
        do
        {
          if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v57 - 8) + 8LL) - 1 <= 1 )
            break;
          v57 -= 8;
        }
        while ( v87 != v57 );
        v95 = v57;
      }
      if ( v86 == v95 )
        goto LABEL_90;
    }
    while ( 1 )
    {
      v32 = v10;
      v10 = *(_QWORD *)(v10 + 8);
      v106 = (__int64 *)(v32 - 24);
      v105 = *(_QWORD *)(v32 - 8) & 0xFFFFFFFFFFFFFFF8LL;
      if ( (*(_QWORD *)(v32 - 8) & 4) != 0 )
        v105 = **(_QWORD **)(*(_QWORD *)(v32 - 8) & 0xFFFFFFFFFFFFFFF8LL);
      if ( *(_BYTE *)(v32 - 16) != 27 )
        break;
      v98 = *(_QWORD *)(v32 + 112);
      v11 = a2(a3, v98);
      v97 = v11;
      v12 = v11;
      if ( v11 )
      {
        v13 = sub_2AC42A0(*a1, *(_QWORD *)(v11 + 16));
        v14 = sub_2C47690(*a1, *(_QWORD *)(v12 + 32), a4);
        v15 = *a1;
        v102 = v14;
        v96 = *a1 + 272;
        v107 = *(_BYTE **)(v32 + 64);
        if ( v107 )
          sub_2C25AB0((__int64 *)&v107);
        v18 = sub_22077B0(0xA8u);
        if ( !v18 )
          goto LABEL_36;
        v108 = v107;
        if ( v107 )
        {
          sub_2C25AB0((__int64 *)&v108);
          v109 = v108;
          if ( v108 )
          {
            sub_2C25AB0((__int64 *)&v109);
            v110 = v109;
            if ( v109 )
            {
              sub_2C25AB0((__int64 *)&v110);
              v117 = v110;
              if ( v110 )
                sub_2C25AB0((__int64 *)&v117);
LABEL_16:
              *(_QWORD *)(v18 + 24) = 0;
              v19 = v18 + 40;
              *(_QWORD *)(v18 + 32) = 0;
              *(_QWORD *)(v18 + 64) = v13;
              *(_QWORD *)v18 = &unk_4A231A8;
              *(_QWORD *)(v18 + 48) = v18 + 64;
              v91 = (const void *)(v18 + 64);
              v89 = v18 + 48;
              *(_QWORD *)(v18 + 40) = &unk_4A23170;
              *(_BYTE *)(v18 + 8) = 33;
              *(_QWORD *)(v18 + 16) = 0;
              *(_QWORD *)(v18 + 56) = 0x200000001LL;
              v20 = *(unsigned int *)(v13 + 24);
              if ( v20 + 1 > (unsigned __int64)*(unsigned int *)(v13 + 28) )
              {
                sub_C8D5F0(v13 + 16, (const void *)(v13 + 32), v20 + 1, 8u, v16, v17);
                v20 = *(unsigned int *)(v13 + 24);
              }
              *(_QWORD *)(*(_QWORD *)(v13 + 16) + 8 * v20) = v19;
              ++*(_DWORD *)(v13 + 24);
              *(_QWORD *)(v18 + 80) = 0;
              *(_QWORD *)(v18 + 40) = &unk_4A23AA8;
              v21 = (__int64)v117;
              *(_QWORD *)v18 = &unk_4A23A70;
              *(_QWORD *)(v18 + 88) = v21;
              if ( v21 )
              {
                sub_2C25AB0((__int64 *)(v18 + 88));
                if ( v117 )
                  sub_B91220((__int64)&v117, (__int64)v117);
              }
              sub_2BF0340(v18 + 96, 1, v98, v18, v16, v17);
              v23 = (__int64)v110;
              *(_QWORD *)v18 = &unk_4A231C8;
              *(_QWORD *)(v18 + 40) = &unk_4A23200;
              *(_QWORD *)(v18 + 96) = &unk_4A23238;
              if ( v23 )
                sub_B91220((__int64)&v110, v23);
              v24 = (__int64)v109;
              *(_QWORD *)v18 = &unk_4A23FE8;
              *(_QWORD *)(v18 + 40) = &unk_4A24030;
              *(_QWORD *)(v18 + 96) = &unk_4A24068;
              if ( v24 )
                sub_B91220((__int64)&v109, v24);
              *(_QWORD *)(v18 + 152) = v97;
              *(_QWORD *)v18 = &unk_4A232E8;
              *(_QWORD *)(v18 + 96) = &unk_4A23370;
              v25 = *(unsigned int *)(v18 + 56);
              *(_QWORD *)(v18 + 40) = &unk_4A23338;
              if ( v25 + 1 > (unsigned __int64)*(unsigned int *)(v18 + 60) )
              {
                sub_C8D5F0(v89, v91, v25 + 1, 8u, v25 + 1, v22);
                v25 = *(unsigned int *)(v18 + 56);
              }
              *(_QWORD *)(*(_QWORD *)(v18 + 48) + 8 * v25) = v102;
              ++*(_DWORD *)(v18 + 56);
              v26 = *(unsigned int *)(v102 + 24);
              if ( v26 + 1 > (unsigned __int64)*(unsigned int *)(v102 + 28) )
              {
                sub_C8D5F0(v102 + 16, (const void *)(v102 + 32), v26 + 1, 8u, v26 + 1, v22);
                v26 = *(unsigned int *)(v102 + 24);
              }
              *(_QWORD *)(*(_QWORD *)(v102 + 16) + 8 * v26) = v19;
              ++*(_DWORD *)(v102 + 24);
              if ( v108 )
                sub_B91220((__int64)&v108, (__int64)v108);
              *(_QWORD *)(v18 + 160) = 0;
              *(_QWORD *)v18 = &unk_4A24088;
              *(_QWORD *)(v18 + 96) = &unk_4A24110;
              v27 = *(unsigned int *)(v18 + 56);
              *(_QWORD *)(v18 + 40) = &unk_4A240D8;
              if ( v27 + 1 > (unsigned __int64)*(unsigned int *)(v18 + 60) )
              {
                sub_C8D5F0(v89, v91, v27 + 1, 8u, v27 + 1, v22);
                v27 = *(unsigned int *)(v18 + 56);
              }
              *(_QWORD *)(*(_QWORD *)(v18 + 48) + 8 * v27) = v96;
              ++*(_DWORD *)(v18 + 56);
              v28 = *(unsigned int *)(v15 + 296);
              if ( v28 + 1 > (unsigned __int64)*(unsigned int *)(v15 + 300) )
              {
                sub_C8D5F0(v15 + 288, (const void *)(v15 + 304), v28 + 1, 8u, v28 + 1, v22);
                v28 = *(unsigned int *)(v15 + 296);
              }
              *(_QWORD *)(*(_QWORD *)(v15 + 288) + 8 * v28) = v19;
              ++*(_DWORD *)(v15 + 296);
LABEL_36:
              if ( v107 )
                sub_B91220((__int64)&v107, (__int64)v107);
LABEL_38:
              sub_2C19D60((_QWORD *)v18, (__int64)v106);
              v29 = *(_QWORD *)(v18 + 16);
              v30 = v29 & 0xFFFFFFFFFFFFFFF8LL;
              if ( (v29 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
              {
                v31 = *(_QWORD *)(v18 + 16) & 0xFFFFFFFFFFFFFFF8LL;
                if ( (v29 & 4) == 0 || *(_DWORD *)(v30 + 8) )
                {
                  if ( ((v29 >> 2) & 1) != 0 )
                  {
                    if ( *(_DWORD *)(v30 + 8) != 1 )
                      goto LABEL_43;
                    v31 = **(_QWORD **)v30;
                  }
                  sub_2BF1250(v105, v31);
                }
              }
LABEL_43:
              sub_2C19E60(v106);
              goto LABEL_44;
            }
LABEL_82:
            v117 = 0;
            goto LABEL_16;
          }
        }
        else
        {
          v109 = 0;
        }
        v110 = 0;
        goto LABEL_82;
      }
LABEL_44:
      if ( v101 == v10 )
        goto LABEL_85;
    }
    v33 = *(unsigned __int8 **)(v105 + 40);
    v34 = *v33;
    switch ( (_BYTE)v34 )
    {
      case '=':
        v35 = **(_QWORD **)(v32 + 24);
        v109 = *(_BYTE **)(v32 + 64);
        if ( v109 )
          sub_2C25AB0((__int64 *)&v109);
        v18 = sub_22077B0(0xA8u);
        if ( v18 )
        {
          v110 = v109;
          if ( v109 )
          {
            sub_2C25AB0((__int64 *)&v110);
            v117 = v110;
            if ( v110 )
              sub_2C25AB0((__int64 *)&v117);
          }
          else
          {
            v117 = 0;
          }
          *(_BYTE *)(v18 + 8) = 20;
          *(_QWORD *)(v18 + 24) = 0;
          *(_QWORD *)(v18 + 32) = 0;
          *(_QWORD *)v18 = &unk_4A231A8;
          *(_QWORD *)(v18 + 16) = 0;
          *(_QWORD *)(v18 + 64) = v35;
          *(_QWORD *)(v18 + 40) = &unk_4A23170;
          *(_QWORD *)(v18 + 48) = v18 + 64;
          *(_QWORD *)(v18 + 56) = 0x200000001LL;
          v37 = *(unsigned int *)(v35 + 24);
          if ( v37 + 1 > (unsigned __int64)*(unsigned int *)(v35 + 28) )
          {
            sub_C8D5F0(v35 + 16, (const void *)(v35 + 32), v37 + 1, 8u, v37 + 1, v36);
            v37 = *(unsigned int *)(v35 + 24);
          }
          *(_QWORD *)(*(_QWORD *)(v35 + 16) + 8 * v37) = v18 + 40;
          ++*(_DWORD *)(v35 + 24);
          *(_QWORD *)(v18 + 80) = 0;
          *(_QWORD *)(v18 + 40) = &unk_4A23AA8;
          v38 = (__int64)v117;
          *(_QWORD *)v18 = &unk_4A23A70;
          *(_QWORD *)(v18 + 88) = v38;
          if ( v38 )
            sub_2C25AB0((__int64 *)(v18 + 88));
          sub_9C6650(&v117);
          *(_QWORD *)(v18 + 96) = v33;
          *(_WORD *)(v18 + 104) = 0;
          *(_BYTE *)(v18 + 106) = 0;
          *(_QWORD *)(v18 + 40) = &unk_4A24740;
          *(_QWORD *)v18 = &unk_4A24708;
          sub_9C6650(&v110);
          sub_2BF0340(v18 + 112, 1, (__int64)v33, v18, v39, v40);
          *(_QWORD *)v18 = &unk_4A24778;
          *(_QWORD *)(v18 + 40) = &unk_4A247B8;
          *(_QWORD *)(v18 + 112) = &unk_4A247F0;
        }
        if ( v109 )
          sub_B91220((__int64)&v109, (__int64)v109);
        goto LABEL_38;
      case '>':
        v41 = *(__int64 **)(v32 + 24);
        v42 = v41[1];
        v43 = *v41;
        v108 = *(_BYTE **)(v32 + 64);
        if ( v108 )
          sub_2C25AB0((__int64 *)&v108);
        v44 = sub_22077B0(0x70u);
        v45 = (__int64 *)&v108;
        v18 = v44;
        if ( v44 )
        {
          v109 = v108;
          if ( v108 )
          {
            sub_2C25AB0((__int64 *)&v109);
            v117 = (_BYTE *)v42;
            v118 = v43;
            v46 = (__int64 *)&v109;
            v110 = v109;
            v45 = (__int64 *)&v108;
            if ( v109 )
            {
              sub_2C25AB0((__int64 *)&v110);
              v46 = (__int64 *)&v109;
              v45 = (__int64 *)&v108;
            }
          }
          else
          {
            v117 = (_BYTE *)v42;
            v46 = (__int64 *)&v109;
            v118 = v43;
            v110 = 0;
          }
          v47 = (const void *)(v18 + 64);
          *(_BYTE *)(v18 + 8) = 22;
          v48 = 0;
          v90 = v18 + 48;
          v49 = v42;
          *(_QWORD *)(v18 + 48) = v18 + 64;
          v50 = v18;
          v51 = (__int64 *)&v117;
          *(_QWORD *)(v18 + 24) = 0;
          *(_QWORD *)v18 = &unk_4A231A8;
          *(_QWORD *)(v18 + 32) = 0;
          *(_QWORD *)(v18 + 16) = 0;
          *(_QWORD *)(v18 + 40) = &unk_4A23170;
          *(_QWORD *)(v18 + 56) = 0x200000000LL;
          v52 = v18 + 40;
          v53 = v49;
          v99 = v33;
          v54 = v52;
          v92 = v47;
          while ( 1 )
          {
            *((_QWORD *)v47 + v48) = v53;
            ++*(_DWORD *)(v50 + 56);
            v55 = *(unsigned int *)(v53 + 24);
            if ( v55 + 1 > (unsigned __int64)*(unsigned int *)(v53 + 28) )
            {
              sub_C8D5F0(v53 + 16, (const void *)(v53 + 32), v55 + 1, 8u, (__int64)v45, (__int64)v46);
              v55 = *(unsigned int *)(v53 + 24);
            }
            ++v51;
            *(_QWORD *)(*(_QWORD *)(v53 + 16) + 8 * v55) = v54;
            ++*(_DWORD *)(v53 + 24);
            if ( v51 == (__int64 *)&v119 )
              break;
            v48 = *(unsigned int *)(v50 + 56);
            v53 = *v51;
            if ( v48 + 1 > (unsigned __int64)*(unsigned int *)(v50 + 60) )
            {
              sub_C8D5F0(v90, v92, v48 + 1, 8u, (__int64)v45, (__int64)v46);
              v48 = *(unsigned int *)(v50 + 56);
            }
            v47 = *(const void **)(v50 + 48);
          }
          *(_QWORD *)(v50 + 80) = 0;
          v18 = v50;
          *(_QWORD *)(v50 + 40) = &unk_4A23AA8;
          v56 = (__int64)v110;
          *(_QWORD *)v50 = &unk_4A23A70;
          *(_QWORD *)(v50 + 88) = v56;
          if ( v56 )
            sub_2C25AB0((__int64 *)(v50 + 88));
          sub_9C6650(&v110);
          *(_QWORD *)(v50 + 96) = v99;
          *(_BYTE *)(v50 + 106) = 0;
          *(_QWORD *)v50 = &unk_4A24708;
          *(_QWORD *)(v50 + 40) = &unk_4A24740;
          *(_WORD *)(v50 + 104) = 0;
          sub_9C6650(&v109);
          *(_QWORD *)v50 = &unk_4A248A8;
          *(_QWORD *)(v50 + 40) = &unk_4A248E8;
        }
        sub_9C6650(&v108);
        goto LABEL_38;
      case '?':
        v58 = *(_QWORD **)(v32 + 24);
        v59 = &v58[*(unsigned int *)(v32 + 32)];
        v60 = sub_22077B0(0xA0u);
        v18 = v60;
        if ( !v60 )
          goto LABEL_38;
        sub_2ABDBC0(v60, 17, v58, v59, v33, v61);
        v62 = (char *)&unk_4A241B8;
        goto LABEL_99;
    }
    if ( (_BYTE)v34 != 85 )
    {
      v68 = *(__int64 **)(v32 + 24);
      if ( (_BYTE)v34 != 86 )
      {
        if ( (unsigned int)(v34 - 67) > 0xC )
        {
          v79 = &v68[*(unsigned int *)(v32 + 32)];
          v80 = sub_22077B0(0xA8u);
          v18 = v80;
          if ( !v80 )
          {
LABEL_138:
            sub_2C19D60(0, (__int64)v106);
            BUG();
          }
          sub_2ABDBC0(v80, 23, v68, v79, v33, v81);
          *(_QWORD *)v18 = &unk_4A23EC8;
          *(_QWORD *)(v18 + 96) = &unk_4A23F38;
          v82 = *v33;
          *(_QWORD *)(v18 + 40) = &unk_4A23F00;
          *(_DWORD *)(v18 + 160) = v82 - 29;
        }
        else
        {
          v73 = *v68;
          v74 = *((_QWORD *)v33 + 1);
          v75 = v34 - 29;
          v104 = v73;
          v76 = sub_22077B0(0xB0u);
          v18 = v76;
          if ( !v76 )
            goto LABEL_138;
          sub_2ABA9E0(v76, 16, v104, v33, v77);
          *(_DWORD *)(v18 + 160) = v75;
          *(_QWORD *)(v18 + 168) = v74;
          *(_QWORD *)v18 = &unk_4A23F58;
          *(_QWORD *)(v18 + 40) = &unk_4A23F90;
          *(_QWORD *)(v18 + 96) = &unk_4A23FC8;
        }
        goto LABEL_38;
      }
      v69 = &v68[*(unsigned int *)(v32 + 32)];
      v70 = sub_22077B0(0xA0u);
      v18 = v70;
      if ( !v70 )
        goto LABEL_38;
      sub_2ABDBC0(v70, 24, v68, v69, v33, v71);
      v62 = (char *)&unk_4A23E20;
LABEL_99:
      *(_QWORD *)v18 = v62 + 16;
      *(_QWORD *)(v18 + 40) = v62 + 80;
      *(_QWORD *)(v18 + 96) = v62 + 136;
      goto LABEL_38;
    }
    v63 = sub_9B78C0(*(_QWORD *)(v105 + 40), a5);
    v64 = *(__int64 **)(v32 + 24);
    v103 = v63;
    v65 = (8LL * *(unsigned int *)(v32 + 32) - 8) >> 3;
    v100 = *((_QWORD *)v33 + 1);
    v110 = (_BYTE *)*((_QWORD *)v33 + 6);
    if ( v110 )
      sub_2C25AB0((__int64 *)&v110);
    v18 = sub_22077B0(0xB8u);
    if ( !v18 )
    {
LABEL_110:
      sub_9C6650(&v110);
      goto LABEL_38;
    }
    v117 = (_BYTE *)*((_QWORD *)v33 + 6);
    if ( v117 )
      sub_2C25AB0((__int64 *)&v117);
    sub_2ABB100(v18, 18, v64, v65, (__int64)v33, (__int64 *)&v117);
    sub_9C6650(&v117);
    *(_QWORD *)v18 = &unk_4A23258;
    *(_QWORD *)(v18 + 96) = &unk_4A232C8;
    v66 = *v33;
    *(_QWORD *)(v18 + 40) = &unk_4A23290;
    if ( (unsigned __int8)(v66 - 82) <= 1u )
    {
      v67 = *((_WORD *)v33 + 1);
      *(_BYTE *)(v18 + 152) = 0;
      *(_DWORD *)(v18 + 156) = v67 & 0x3F;
LABEL_109:
      *(_QWORD *)v18 = &unk_4A23D28;
      *(_QWORD *)(v18 + 96) = &unk_4A23DA0;
      *(_QWORD *)(v18 + 40) = &unk_4A23D68;
      *(_DWORD *)(v18 + 160) = v103;
      *(_QWORD *)(v18 + 168) = v100;
      *(_BYTE *)(v18 + 176) = sub_B46420((__int64)v33);
      *(_BYTE *)(v18 + 177) = sub_B46490((__int64)v33);
      *(_BYTE *)(v18 + 178) = sub_B46970(v33);
      goto LABEL_110;
    }
    if ( (_BYTE)v66 == 58 )
    {
      *(_BYTE *)(v18 + 152) = 2;
LABEL_116:
      v72 = v33[1] >> 1;
LABEL_117:
      *(_BYTE *)(v18 + 156) = v72 & 1 | *(_BYTE *)(v18 + 156) & 0xFE;
      goto LABEL_109;
    }
    if ( (unsigned __int8)v66 > 0x36u )
    {
      if ( (unsigned __int8)(v66 - 55) > 1u )
      {
        if ( (_BYTE)v66 == 63 )
        {
          *(_BYTE *)(v18 + 152) = 4;
          *(_DWORD *)(v18 + 156) = sub_B4DE20((__int64)v33);
          goto LABEL_109;
        }
LABEL_131:
        if ( (((_BYTE)v66 - 68) & 0xFB) != 0 )
        {
          if ( (unsigned __int8)sub_920620((__int64)v33) )
          {
            v83 = v33[1];
            *(_BYTE *)(v18 + 152) = 5;
            v84 = v83 >> 1;
            if ( v84 == 127 )
              v84 = -1;
            LODWORD(v117) = v84;
            sub_2C1AC80(&v109, &v117);
            *(_BYTE *)(v18 + 156) = (_BYTE)v109;
          }
          else
          {
            *(_BYTE *)(v18 + 152) = 7;
            *(_DWORD *)(v18 + 156) = 0;
          }
          goto LABEL_109;
        }
        *(_BYTE *)(v18 + 152) = 6;
        v72 = sub_B44910((__int64)v33);
        goto LABEL_117;
      }
    }
    else
    {
      v78 = 0x40540000000000LL;
      if ( _bittest64(&v78, v66) )
      {
        *(_BYTE *)(v18 + 152) = 1;
        *(_BYTE *)(v18 + 156) = *(_BYTE *)(v18 + 156) & 0xFC | ((v33[1] & 2) != 0) | (v33[1] >> 1) & 2;
        goto LABEL_109;
      }
      if ( (unsigned int)(unsigned __int8)v66 - 48 > 1 )
        goto LABEL_131;
    }
    *(_BYTE *)(v18 + 152) = 3;
    goto LABEL_116;
  }
LABEL_90:
  if ( v114 != v116 )
    _libc_free((unsigned __int64)v114);
}
