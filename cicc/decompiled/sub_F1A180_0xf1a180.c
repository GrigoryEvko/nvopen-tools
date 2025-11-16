// Function: sub_F1A180
// Address: 0xf1a180
//
__int64 __fastcall sub_F1A180(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  __int64 *v8; // rax
  __int64 v9; // rbx
  __int64 *v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r13
  __int64 v13; // rdx
  __int64 v14; // r15
  __int64 v15; // r14
  __int64 v16; // rbx
  __int64 v17; // r13
  __int64 *v18; // rax
  __int64 *v19; // rdx
  __int64 v20; // r12
  int v21; // ebx
  unsigned __int64 v22; // rdx
  __int64 v23; // rdx
  unsigned __int64 v24; // rcx
  unsigned __int64 v25; // rax
  _BYTE *v26; // r12
  _BYTE *v27; // rbx
  __int64 v28; // r14
  unsigned int v29; // r12d
  __int64 v30; // rsi
  __int64 *v32; // rax
  __int64 *v33; // rbx
  __int64 v34; // r15
  int v35; // edx
  __int64 v36; // rax
  __int64 *v37; // r11
  __int64 *v38; // r15
  _BYTE *v39; // r12
  char v40; // al
  int v41; // r10d
  _QWORD *v42; // rdx
  unsigned int v43; // edi
  _QWORD *v44; // rax
  __int64 v45; // rcx
  __int64 *v46; // r13
  __int64 v47; // rax
  __int64 v48; // rdx
  __int64 v49; // rdx
  __int64 v50; // rax
  unsigned __int64 v51; // rdx
  _BYTE *v52; // rax
  _QWORD *v53; // rdx
  __int64 v54; // rcx
  __int64 v55; // r8
  __int64 v56; // r9
  _QWORD *v57; // rdx
  __int64 v58; // rcx
  __int64 v59; // r8
  __int64 v60; // r9
  _BYTE *v61; // rdi
  unsigned __int64 v62; // rax
  char v63; // dl
  unsigned __int8 *v64; // rcx
  _QWORD *v65; // rdx
  int v66; // ecx
  __int64 v67; // rax
  __int64 v68; // r8
  int v69; // r10d
  _QWORD *v70; // r9
  _QWORD *v71; // r8
  __int64 v72; // r13
  int v73; // r9d
  _BYTE *v74; // rsi
  __int64 v75; // rax
  __int64 *v76; // rbx
  int v77; // ecx
  __int64 v78; // r12
  __int64 v79; // rdx
  __int64 v80; // rax
  __int64 v81; // rdx
  __int64 v82; // rax
  __int64 v83; // rdx
  unsigned __int8 v84; // al
  __int64 *v85; // rax
  char *v86; // rax
  char *v87; // rdx
  char *v88; // rax
  char *v89; // rdx
  __int64 v90; // [rsp+0h] [rbp-6C0h]
  __int64 v91; // [rsp+10h] [rbp-6B0h]
  __int64 v92; // [rsp+20h] [rbp-6A0h]
  __int64 v93; // [rsp+28h] [rbp-698h]
  __int64 v94; // [rsp+30h] [rbp-690h]
  __int64 v95; // [rsp+38h] [rbp-688h]
  __int64 v96; // [rsp+40h] [rbp-680h]
  _QWORD *v97; // [rsp+48h] [rbp-678h]
  unsigned __int8 v98; // [rsp+5Fh] [rbp-661h] BYREF
  __int64 v99[2]; // [rsp+60h] [rbp-660h] BYREF
  __int64 v100[2]; // [rsp+70h] [rbp-650h] BYREF
  __int64 v101; // [rsp+80h] [rbp-640h] BYREF
  __int64 v102; // [rsp+88h] [rbp-638h]
  __int64 v103; // [rsp+90h] [rbp-630h]
  unsigned int v104; // [rsp+98h] [rbp-628h]
  __int64 v105; // [rsp+A0h] [rbp-620h] BYREF
  char *v106; // [rsp+A8h] [rbp-618h]
  __int64 v107; // [rsp+B0h] [rbp-610h]
  int v108; // [rsp+B8h] [rbp-608h]
  char v109; // [rsp+BCh] [rbp-604h]
  char v110; // [rsp+C0h] [rbp-600h] BYREF
  __int64 v111; // [rsp+100h] [rbp-5C0h] BYREF
  char *v112; // [rsp+108h] [rbp-5B8h]
  __int64 v113; // [rsp+110h] [rbp-5B0h]
  int v114; // [rsp+118h] [rbp-5A8h]
  char v115; // [rsp+11Ch] [rbp-5A4h]
  char v116; // [rsp+120h] [rbp-5A0h] BYREF
  __int64 v117; // [rsp+160h] [rbp-560h] BYREF
  __int64 *v118; // [rsp+168h] [rbp-558h]
  __int64 v119; // [rsp+170h] [rbp-550h]
  int v120; // [rsp+178h] [rbp-548h]
  char v121; // [rsp+17Ch] [rbp-544h]
  char v122; // [rsp+180h] [rbp-540h] BYREF
  _BYTE *v123; // [rsp+280h] [rbp-440h] BYREF
  __int64 v124; // [rsp+288h] [rbp-438h]
  _BYTE v125[1072]; // [rsp+290h] [rbp-430h] BYREF

  v6 = a1;
  v118 = (__int64 *)&v122;
  v123 = v125;
  v124 = 0x8000000000LL;
  v106 = &v110;
  v112 = &v116;
  v99[0] = a1;
  v90 = a2;
  v98 = 0;
  v117 = 0;
  v119 = 32;
  v120 = 0;
  v121 = 1;
  v101 = 0;
  v102 = 0;
  v103 = 0;
  v104 = 0;
  v105 = 0;
  v107 = 8;
  v108 = 0;
  v109 = 1;
  v111 = 0;
  v113 = 8;
  v114 = 0;
  v115 = 1;
  v99[1] = (__int64)&v98;
  v8 = *(__int64 **)(a1 + 232);
  v91 = *v8;
  v92 = *v8 + 8LL * *((unsigned int *)v8 + 2);
  if ( *v8 != v92 )
  {
    while ( 1 )
    {
      v9 = *(_QWORD *)(v92 - 8);
      v95 = v9;
      if ( !sub_AA5B70(v9) )
      {
        v12 = *(_QWORD *)(v9 + 16);
        if ( !v12 )
          goto LABEL_9;
        while ( 1 )
        {
          v13 = *(_QWORD *)(v12 + 24);
          if ( (unsigned __int8)(*(_BYTE *)v13 - 30) <= 0xAu )
            break;
          v12 = *(_QWORD *)(v12 + 8);
          if ( !v12 )
            goto LABEL_9;
        }
LABEL_7:
        v14 = *(_QWORD *)(v13 + 40);
        v100[0] = v14;
        v100[1] = v9;
        if ( sub_F11A70(a1 + 248, v100) || (a2 = v9, (unsigned __int8)sub_B19720(*(_QWORD *)(a1 + 80), v9, v14)) )
        {
          while ( 1 )
          {
            v12 = *(_QWORD *)(v12 + 8);
            if ( !v12 )
              goto LABEL_9;
            v13 = *(_QWORD *)(v12 + 24);
            if ( (unsigned __int8)(*(_BYTE *)v13 - 30) <= 0xAu )
              goto LABEL_7;
          }
        }
      }
      if ( !v121 )
        break;
      v32 = v118;
      v11 = HIDWORD(v119);
      v10 = &v118[HIDWORD(v119)];
      if ( v118 == v10 )
      {
LABEL_140:
        if ( HIDWORD(v119) >= (unsigned int)v119 )
          break;
        ++HIDWORD(v119);
        *v10 = v9;
        ++v117;
      }
      else
      {
        a2 = v9;
        while ( v9 != *v32 )
        {
          if ( v10 == ++v32 )
            goto LABEL_140;
        }
      }
LABEL_47:
      v94 = v9 + 48;
      v96 = *(_QWORD *)(v9 + 56);
      if ( v96 != v9 + 48 )
      {
        while ( 1 )
        {
          v34 = v96;
          v35 = *(_DWORD *)(v96 - 20);
          v96 = *(_QWORD *)(v96 + 8);
          v97 = (_QWORD *)(v34 - 24);
          v36 = v35 & 0x7FFFFFF;
          if ( *(_QWORD *)(v34 - 8) )
          {
            if ( (_DWORD)v36 )
            {
              if ( (*(_BYTE *)(v34 - 17) & 0x40) != 0 )
              {
                v33 = *(__int64 **)(v34 - 32);
                if ( *(_BYTE *)*v33 > 0x15u )
                  goto LABEL_113;
              }
              else
              {
                v33 = &v97[-4 * v36];
                if ( *(_BYTE *)*v33 > 0x15u )
                  goto LABEL_59;
              }
            }
            a2 = sub_97D880((__int64)v97, *(_BYTE **)(a1 + 88), *(__int64 **)(a1 + 72));
            if ( a2 )
            {
              sub_BD84D0((__int64)v97, a2);
              a2 = *(_QWORD *)(a1 + 72);
              if ( (unsigned __int8)sub_F50EE0(v97, a2) )
                sub_B43D60(v97);
              v98 = 1;
              goto LABEL_54;
            }
            v35 = *(_DWORD *)(v34 - 20);
          }
          v36 = v35 & 0x7FFFFFF;
          if ( (*(_BYTE *)(v34 - 17) & 0x40) != 0 )
          {
            v33 = *(__int64 **)(v34 - 32);
LABEL_113:
            v37 = &v33[4 * v36];
            goto LABEL_60;
          }
          v33 = &v97[-4 * v36];
LABEL_59:
          v37 = (__int64 *)(v34 - 24);
LABEL_60:
          if ( v33 != v37 )
          {
            v93 = v34;
            v38 = v37;
            while ( 1 )
            {
              v39 = (_BYTE *)*v33;
              v40 = *(_BYTE *)*v33;
              if ( v40 != 11 && v40 != 5 )
                goto LABEL_63;
              a2 = v104;
              if ( !v104 )
                break;
              v41 = 1;
              v42 = 0;
              v43 = (v104 - 1) & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
              v44 = (_QWORD *)(v102 + 16LL * v43);
              v45 = *v44;
              if ( v39 == (_BYTE *)*v44 )
              {
LABEL_67:
                v46 = v44 + 1;
                v47 = v44[1];
                if ( !v47 )
                  goto LABEL_107;
                if ( (_BYTE *)v47 != v39 )
                  goto LABEL_69;
LABEL_63:
                v33 += 4;
                if ( v38 == v33 )
                  goto LABEL_76;
              }
              else
              {
                while ( v45 != -4096 )
                {
                  if ( v45 == -8192 && !v42 )
                    v42 = v44;
                  v43 = (v104 - 1) & (v41 + v43);
                  v44 = (_QWORD *)(v102 + 16LL * v43);
                  v45 = *v44;
                  if ( v39 == (_BYTE *)*v44 )
                    goto LABEL_67;
                  ++v41;
                }
                if ( !v42 )
                  v42 = v44;
                ++v101;
                v66 = v103 + 1;
                if ( 4 * ((int)v103 + 1) < 3 * v104 )
                {
                  if ( v104 - HIDWORD(v103) - v66 <= v104 >> 3 )
                  {
                    sub_F19020((__int64)&v101, v104);
                    if ( !v104 )
                    {
LABEL_178:
                      LODWORD(v103) = v103 + 1;
                      BUG();
                    }
                    v71 = 0;
                    LODWORD(v72) = (v104 - 1) & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
                    v73 = 1;
                    v66 = v103 + 1;
                    v42 = (_QWORD *)(v102 + 16LL * (unsigned int)v72);
                    v74 = (_BYTE *)*v42;
                    if ( v39 != (_BYTE *)*v42 )
                    {
                      while ( v74 != (_BYTE *)-4096LL )
                      {
                        if ( !v71 && v74 == (_BYTE *)-8192LL )
                          v71 = v42;
                        v72 = (v104 - 1) & ((_DWORD)v72 + v73);
                        v42 = (_QWORD *)(v102 + 16 * v72);
                        v74 = (_BYTE *)*v42;
                        if ( v39 == (_BYTE *)*v42 )
                          goto LABEL_104;
                        ++v73;
                      }
                      if ( v71 )
                        v42 = v71;
                    }
                  }
                  goto LABEL_104;
                }
LABEL_116:
                sub_F19020((__int64)&v101, 2 * v104);
                if ( !v104 )
                  goto LABEL_178;
                LODWORD(v67) = (v104 - 1) & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
                v66 = v103 + 1;
                v42 = (_QWORD *)(v102 + 16LL * (unsigned int)v67);
                v68 = *v42;
                if ( v39 != (_BYTE *)*v42 )
                {
                  v69 = 1;
                  v70 = 0;
                  while ( v68 != -4096 )
                  {
                    if ( v68 == -8192 && !v70 )
                      v70 = v42;
                    v67 = (v104 - 1) & ((_DWORD)v67 + v69);
                    v42 = (_QWORD *)(v102 + 16 * v67);
                    v68 = *v42;
                    if ( v39 == (_BYTE *)*v42 )
                      goto LABEL_104;
                    ++v69;
                  }
                  if ( v70 )
                    v42 = v70;
                }
LABEL_104:
                LODWORD(v103) = v66;
                if ( *v42 != -4096 )
                  --HIDWORD(v103);
                *v42 = v39;
                v46 = v42 + 1;
                v42[1] = 0;
LABEL_107:
                a2 = *(_QWORD *)(a1 + 88);
                v47 = sub_97B670(v39, a2, *(_QWORD *)(a1 + 72));
                *v46 = v47;
                if ( (_BYTE *)v47 == v39 )
                  goto LABEL_63;
                if ( *v33 )
                {
LABEL_69:
                  v48 = v33[1];
                  *(_QWORD *)v33[2] = v48;
                  if ( v48 )
                    *(_QWORD *)(v48 + 16) = v33[2];
                }
                *v33 = v47;
                if ( v47 )
                {
                  v49 = *(_QWORD *)(v47 + 16);
                  v33[1] = v49;
                  if ( v49 )
                  {
                    a2 = (__int64)(v33 + 1);
                    *(_QWORD *)(v49 + 16) = v33 + 1;
                  }
                  v33[2] = v47 + 16;
                  *(_QWORD *)(v47 + 16) = v33;
                }
                v33 += 4;
                v98 = 1;
                if ( v38 == v33 )
                {
LABEL_76:
                  v34 = v93;
                  goto LABEL_77;
                }
              }
            }
            ++v101;
            goto LABEL_116;
          }
LABEL_77:
          if ( sub_B46AA0((__int64)v97) )
            goto LABEL_54;
          v50 = (unsigned int)v124;
          v51 = (unsigned int)v124 + 1LL;
          if ( v51 > HIDWORD(v124) )
          {
            a2 = (__int64)v125;
            sub_C8D5F0((__int64)&v123, v125, v51, 8u, a5, a6);
            v50 = (unsigned int)v124;
          }
          *(_QWORD *)&v123[8 * v50] = v97;
          LODWORD(v124) = v124 + 1;
          if ( (*(_BYTE *)(v34 - 17) & 0x20) == 0 )
          {
LABEL_54:
            if ( v94 == v96 )
              break;
          }
          else
          {
            v52 = (_BYTE *)sub_B91C10((__int64)v97, 7);
            sub_F09A90(v52, (__int64)&v105, v53, v54, v55, v56);
            v61 = 0;
            if ( (*(_BYTE *)(v34 - 17) & 0x20) != 0 )
              v61 = (_BYTE *)sub_B91C10((__int64)v97, 8);
            a2 = (__int64)&v111;
            sub_F09A90(v61, (__int64)&v111, v57, v58, v59, v60);
            if ( v94 == v96 )
              break;
          }
        }
      }
      v62 = *(_QWORD *)(v95 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v94 == v62 )
        goto LABEL_180;
      if ( !v62 )
        BUG();
      if ( (unsigned int)*(unsigned __int8 *)(v62 - 24) - 30 > 0xA )
LABEL_180:
        BUG();
      v63 = *(_BYTE *)(v62 - 24);
      if ( v63 == 31 )
      {
        if ( (*(_DWORD *)(v62 - 20) & 0x7FFFFFF) != 3 )
          goto LABEL_10;
        v64 = *(unsigned __int8 **)(v62 - 120);
        a2 = *v64;
        if ( (unsigned int)(a2 - 12) > 1 )
        {
          if ( (_BYTE)a2 == 17 )
          {
            v65 = (_QWORD *)*((_QWORD *)v64 + 3);
            if ( *((_DWORD *)v64 + 8) > 0x40u )
              v65 = (_QWORD *)*v65;
            a2 = v95;
            sub_F19BD0(v99, v95, *(_QWORD *)(v62 - 32LL * (v65 == 0) - 56));
          }
          goto LABEL_10;
        }
      }
      else
      {
        if ( v63 != 32 )
          goto LABEL_10;
        v76 = *(__int64 **)(v62 - 32);
        a5 = *v76;
        v77 = *(unsigned __int8 *)*v76;
        if ( (unsigned int)(v77 - 12) > 1 )
        {
          if ( (_BYTE)v77 == 17 )
          {
            v78 = ((*(_DWORD *)(v62 - 20) & 0x7FFFFFFu) >> 1) - 1;
            sub_F08550(v62 - 24, 0, v62 - 24, v78, a5);
            v80 = v79;
            v81 = 4;
            if ( v80 != v78 && (unsigned int)v80 != 4294967294LL )
              v81 = 4LL * (unsigned int)(2 * v80 + 3);
            a2 = v95;
            sub_F19BD0(v99, v95, v76[v81]);
          }
          goto LABEL_10;
        }
      }
LABEL_9:
      a2 = v95;
      sub_F19BD0(v99, v95, 0);
LABEL_10:
      v92 -= 8;
      if ( v91 == v92 )
      {
        v6 = a1;
        goto LABEL_12;
      }
    }
    a2 = v9;
    sub_C8CC70((__int64)&v117, v9, (__int64)v10, v11, a5, a6);
    goto LABEL_47;
  }
LABEL_12:
  v15 = *(_QWORD *)(v90 + 80);
  v16 = v90 + 72;
  if ( v90 + 72 != v15 )
  {
    while ( 1 )
    {
      v17 = v15 - 24;
      if ( !v15 )
        v17 = 0;
      if ( v121 )
      {
        v18 = v118;
        v19 = &v118[HIDWORD(v119)];
        if ( v118 == v19 )
          goto LABEL_130;
        while ( v17 != *v18 )
        {
          if ( v19 == ++v18 )
            goto LABEL_130;
        }
LABEL_20:
        v15 = *(_QWORD *)(v15 + 8);
        if ( v16 == v15 )
          break;
      }
      else
      {
        a2 = v17;
        if ( sub_C8CA60((__int64)&v117, v17) )
          goto LABEL_20;
LABEL_130:
        v75 = sub_F55A40(v17);
        v15 = *(_QWORD *)(v15 + 8);
        v98 |= (_DWORD)v75 + HIDWORD(v75) != 0;
        if ( v16 == v15 )
          break;
      }
    }
  }
  v20 = *(_QWORD *)(v6 + 40);
  v21 = v124;
  v22 = (unsigned int)v124 + 16LL;
  if ( v22 > *(unsigned int *)(v20 + 12) )
  {
    a2 = v20 + 16;
    sub_C8D5F0(*(_QWORD *)(v6 + 40), (const void *)(v20 + 16), v22, 8u, a5, a6);
  }
  v23 = *(_QWORD *)(v20 + 2064) + 1LL;
  if ( v21 )
  {
    *(_QWORD *)(v20 + 2064) = v23;
    v24 = ((((((4 * v21 / 3u + 1) | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 2)
           | (4 * v21 / 3u + 1)
           | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 4)
         | (((4 * v21 / 3u + 1) | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 2)
         | (4 * v21 / 3u + 1)
         | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 8;
    v25 = ((v24
          | (((((4 * v21 / 3u + 1) | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 2)
            | (4 * v21 / 3u + 1)
            | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 4)
          | (((4 * v21 / 3u + 1) | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 2)
          | (4 * v21 / 3u + 1)
          | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 16)
        | v24
        | (((((4 * v21 / 3u + 1) | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 2)
          | (4 * v21 / 3u + 1)
          | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 4)
        | (((4 * v21 / 3u + 1) | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 2)
        | (4 * v21 / 3u + 1)
        | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1);
    a2 = v25 + 1;
    if ( *(_DWORD *)(v20 + 2088) < (unsigned int)(v25 + 1) )
      sub_9BAAD0(v20 + 2064, a2);
  }
  else
  {
    *(_QWORD *)(v20 + 2064) = v23;
  }
  v26 = v123;
  v27 = &v123[8 * (unsigned int)v124];
  if ( v123 != v27 )
  {
    do
    {
      while ( 1 )
      {
        v28 = *((_QWORD *)v27 - 1);
        a2 = *(_QWORD *)(v6 + 72);
        if ( !(unsigned __int8)sub_F50EE0(v28, a2) )
          break;
LABEL_31:
        v27 -= 8;
        sub_F54ED0(v28);
        sub_B43D60((_QWORD *)v28);
        v98 = 1;
        if ( v26 == v27 )
          goto LABEL_32;
      }
      if ( *(_BYTE *)v28 == 85 )
      {
        v82 = *(_QWORD *)(v28 - 32);
        if ( v82 )
        {
          if ( !*(_BYTE *)v82
            && *(_QWORD *)(v82 + 24) == *(_QWORD *)(v28 + 80)
            && (*(_BYTE *)(v82 + 33) & 0x20) != 0
            && *(_DWORD *)(v82 + 36) == 155 )
          {
            v83 = *(_QWORD *)(*(_QWORD *)(v28 - 32LL * (*(_DWORD *)(v28 + 4) & 0x7FFFFFF)) + 24LL);
            v84 = *(_BYTE *)(v83 - 16);
            if ( (v84 & 2) != 0 )
              v85 = *(__int64 **)(v83 - 32);
            else
              v85 = (__int64 *)(v83 - 8LL * ((v84 >> 2) & 0xF) - 16);
            a2 = *v85;
            if ( (unsigned __int8)(*(_BYTE *)*v85 - 5) > 0x1Fu )
              goto LABEL_31;
            if ( v109 )
            {
              v86 = v106;
              v87 = &v106[8 * HIDWORD(v107)];
              if ( v106 == v87 )
                goto LABEL_31;
              while ( a2 != *(_QWORD *)v86 )
              {
                v86 += 8;
                if ( v87 == v86 )
                  goto LABEL_31;
              }
            }
            else if ( !sub_C8CA60((__int64)&v105, a2) )
            {
              goto LABEL_31;
            }
            if ( v115 )
            {
              v88 = v112;
              v89 = &v112[8 * HIDWORD(v113)];
              if ( v112 == v89 )
                goto LABEL_31;
              while ( a2 != *(_QWORD *)v88 )
              {
                v88 += 8;
                if ( v89 == v88 )
                  goto LABEL_31;
              }
            }
            else if ( !sub_C8CA60((__int64)&v111, a2) )
            {
              goto LABEL_31;
            }
          }
        }
      }
      a2 = v28;
      v27 -= 8;
      sub_F15FC0(*(_QWORD *)(v6 + 40), v28);
    }
    while ( v26 != v27 );
  }
LABEL_32:
  v29 = v98;
  if ( v115 )
  {
    if ( v109 )
      goto LABEL_34;
  }
  else
  {
    _libc_free(v112, a2);
    if ( v109 )
      goto LABEL_34;
  }
  _libc_free(v106, a2);
LABEL_34:
  v30 = 16LL * v104;
  sub_C7D6A0(v102, v30, 8);
  if ( v123 != v125 )
    _libc_free(v123, v30);
  if ( !v121 )
    _libc_free(v118, v30);
  return v29;
}
