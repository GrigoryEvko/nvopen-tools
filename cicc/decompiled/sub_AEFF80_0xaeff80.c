// Function: sub_AEFF80
// Address: 0xaeff80
//
__int64 __fastcall sub_AEFF80(__int64 a1)
{
  _QWORD *v1; // r14
  __int64 v2; // rax
  _QWORD *v3; // rcx
  __int64 v4; // rdi
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // r12
  __int64 v8; // rbx
  _QWORD *v9; // rdi
  unsigned __int64 v10; // rsi
  __int64 v11; // rax
  __int64 v12; // rbx
  __int64 v13; // r12
  __int64 v14; // rdi
  __int64 v15; // rsi
  __int64 v16; // rax
  _QWORD *v17; // rbx
  _QWORD *v18; // r12
  __int64 v19; // rdi
  __int64 v20; // rsi
  __int64 v21; // rax
  _QWORD *v22; // rbx
  _QWORD *v23; // r12
  __int64 v24; // rdi
  __int64 v26; // rax
  __int64 v27; // rax
  _BYTE *v28; // rax
  _BYTE *v29; // r13
  int v30; // ecx
  unsigned int v31; // r8d
  _QWORD *v32; // rdx
  _QWORD *v33; // rax
  _BYTE *v34; // rdi
  _QWORD *v35; // rdi
  unsigned int v36; // esi
  int v37; // ecx
  unsigned int v38; // r8d
  __int64 *v39; // rdx
  __int64 *v40; // rax
  _BYTE *v41; // rdi
  __int64 v42; // rax
  __int64 v43; // rdx
  __int64 v44; // rax
  _BYTE *v45; // rax
  __int64 v46; // rcx
  int v47; // r8d
  unsigned int v48; // edx
  __int64 v49; // rdi
  __int64 v50; // rax
  __int64 v51; // r9
  __int64 v52; // rdi
  _QWORD *v53; // rbx
  int v54; // r11d
  __int64 v55; // rcx
  int v56; // edx
  __int64 *v57; // rax
  __int64 v58; // rcx
  __int64 *v59; // rdx
  unsigned int v60; // ecx
  int v61; // edx
  _BYTE *v62; // r11
  int v63; // edi
  __int64 *v64; // rsi
  int v65; // esi
  unsigned int v66; // r11d
  _QWORD *v67; // rcx
  __int64 v68; // r8
  unsigned int v69; // ecx
  __int64 v70; // r11
  int v71; // edi
  _QWORD *v72; // rsi
  int v73; // esi
  unsigned int v74; // r11d
  __int64 *v75; // rcx
  _BYTE *v76; // r8
  __int64 v77; // rcx
  _QWORD *v78; // rax
  _QWORD *v79; // rbx
  __int64 v80; // rdi
  _QWORD *v81; // r15
  __int64 v82; // rdi
  _QWORD *v83; // rax
  __int64 v84; // rdx
  _QWORD *v85; // r12
  __int64 v86; // rdi
  _QWORD *v87; // r13
  _QWORD *v88; // rax
  _QWORD *v89; // rbx
  __int64 v90; // rdi
  _QWORD *v91; // r13
  __int64 v92; // rax
  __int64 v93; // rax
  _QWORD *v94; // rax
  _QWORD *v95; // r12
  __int64 v96; // rdi
  _QWORD *v97; // r15
  __int64 v98; // rax
  _QWORD *v99; // rax
  _QWORD *v100; // rax
  int v101; // edx
  int v102; // r10d
  __int64 v103; // rsi
  unsigned int v105; // [rsp+8h] [rbp-108h]
  unsigned int v106; // [rsp+8h] [rbp-108h]
  __int64 v107; // [rsp+8h] [rbp-108h]
  _QWORD *v108; // [rsp+10h] [rbp-100h]
  _QWORD *v109; // [rsp+18h] [rbp-F8h]
  __int64 v110; // [rsp+20h] [rbp-F0h]
  _BYTE *v111; // [rsp+30h] [rbp-E0h] BYREF
  __int64 v112; // [rsp+38h] [rbp-D8h] BYREF
  __int64 v113; // [rsp+40h] [rbp-D0h] BYREF
  _QWORD *v114; // [rsp+48h] [rbp-C8h]
  __int64 v115; // [rsp+50h] [rbp-C0h]
  unsigned int v116; // [rsp+58h] [rbp-B8h]
  __int64 v117; // [rsp+60h] [rbp-B0h] BYREF
  _QWORD *v118; // [rsp+68h] [rbp-A8h]
  __int64 v119; // [rsp+70h] [rbp-A0h]
  unsigned int v120; // [rsp+78h] [rbp-98h]
  __int64 v121; // [rsp+80h] [rbp-90h] BYREF
  __int64 v122; // [rsp+88h] [rbp-88h]
  __int64 v123; // [rsp+90h] [rbp-80h]
  unsigned int v124; // [rsp+98h] [rbp-78h]
  __m128i v125; // [rsp+A0h] [rbp-70h] BYREF
  _BYTE v126[96]; // [rsp+B0h] [rbp-60h] BYREF

  v2 = sub_B2BEC0(a1);
  v3 = *(_QWORD **)(a1 + 80);
  v113 = 0;
  v110 = v2;
  v114 = 0;
  v115 = 0;
  v116 = 0;
  v117 = 0;
  v118 = 0;
  v119 = 0;
  v120 = 0;
  v121 = 0;
  v122 = 0;
  v123 = 0;
  v124 = 0;
  v108 = v3;
  v109 = (_QWORD *)(a1 + 72);
  if ( v3 != (_QWORD *)(a1 + 72) )
  {
    while ( 1 )
    {
      if ( !v108 )
        BUG();
      v1 = (_QWORD *)v108[4];
      if ( v1 != v108 + 3 )
        break;
LABEL_17:
      v108 = (_QWORD *)v108[1];
      if ( v109 == v108 )
      {
        v9 = *(_QWORD **)(a1 + 80);
        goto LABEL_19;
      }
    }
    while ( 1 )
    {
      if ( !v1 )
        BUG();
      v4 = v1[5];
      if ( v4 )
      {
        v5 = sub_B14240(v4);
        v7 = v6;
        v8 = v5;
        if ( v5 != v6 )
        {
          while ( *(_BYTE *)(v8 + 32) )
          {
            v8 = *(_QWORD *)(v8 + 8);
            if ( v8 == v6 )
              goto LABEL_15;
          }
          if ( v8 != v6 )
            break;
        }
      }
LABEL_15:
      if ( *((_BYTE *)v1 - 24) == 85 )
      {
        v42 = *(v1 - 7);
        if ( v42 )
        {
          if ( !*(_BYTE *)v42
            && *(_QWORD *)(v42 + 24) == v1[7]
            && (*(_BYTE *)(v42 + 33) & 0x20) != 0
            && *(_DWORD *)(v42 + 36) == 69 )
          {
            v43 = *(_QWORD *)(v1[4 * (2LL - (*((_DWORD *)v1 - 5) & 0x7FFFFFF)) - 3] + 24LL);
            if ( !(unsigned int)((__int64)(*(_QWORD *)(v43 + 24) - *(_QWORD *)(v43 + 16)) >> 3) )
            {
              if ( sub_B58EB0(v1 - 3, 0) )
              {
                v44 = sub_B58EB0(v1 - 3, 0);
                v45 = (_BYTE *)sub_BD3990(v44);
                if ( *v45 == 60 )
                {
                  v111 = v45;
                  if ( (unsigned __int8)sub_B4D040(v45) )
                  {
                    sub_B4CED0(&v125, v111, v110);
                    if ( !v126[0] || !v125.m128i_i8[8] )
                    {
                      if ( v116 )
                      {
                        v46 = (__int64)v111;
                        v47 = 1;
                        v48 = (v116 - 1) & (((unsigned int)v111 >> 9) ^ ((unsigned int)v111 >> 4));
                        v49 = 0;
                        v50 = (__int64)&v114[7 * v48];
                        v51 = *(_QWORD *)v50;
                        if ( v111 == *(_BYTE **)v50 )
                        {
LABEL_79:
                          v52 = v50 + 8;
LABEL_80:
                          sub_AE6EC0(v52, (__int64)(v1 - 3));
                          v53 = sub_AEC7A0((__int64)&v121, (__int64 *)&v111);
                          v125.m128i_i64[0] = *(_QWORD *)(v1[4 * (1LL - (*((_DWORD *)v1 - 5) & 0x7FFFFFF)) - 3] + 24LL);
                          sub_AE7A80((__int64)&v112, (__int64)(v1 - 3));
                          v125.m128i_i64[1] = sub_B10CD0(&v112);
                          if ( v112 )
                            sub_B91220(&v112);
                          sub_AEFB90((__int64)v53, &v125);
                          goto LABEL_16;
                        }
                        while ( v51 != -4096 )
                        {
                          if ( v51 == -8192 && !v49 )
                            v49 = v50;
                          v48 = (v116 - 1) & (v48 + v47);
                          v50 = (__int64)&v114[7 * v48];
                          v51 = *(_QWORD *)v50;
                          if ( v111 == *(_BYTE **)v50 )
                            goto LABEL_79;
                          ++v47;
                        }
                        if ( v49 )
                          v50 = v49;
                        ++v113;
                        v101 = v115 + 1;
                        v125.m128i_i64[0] = v50;
                        if ( 4 * ((int)v115 + 1) < 3 * v116 )
                        {
                          if ( v116 - HIDWORD(v115) - v101 <= v116 >> 3 )
                          {
                            sub_AECA20((__int64)&v113, v116);
                            goto LABEL_216;
                          }
                          goto LABEL_217;
                        }
                      }
                      else
                      {
                        ++v113;
                        v125.m128i_i64[0] = 0;
                      }
                      sub_AECA20((__int64)&v113, 2 * v116);
LABEL_216:
                      sub_AEA950((__int64)&v113, (__int64 *)&v111, &v125);
                      v46 = (__int64)v111;
                      v101 = v115 + 1;
                      v50 = v125.m128i_i64[0];
LABEL_217:
                      LODWORD(v115) = v101;
                      if ( *(_QWORD *)v50 != -4096 )
                        --HIDWORD(v115);
                      *(_QWORD *)v50 = v46;
                      v52 = v50 + 8;
                      *(_QWORD *)(v50 + 8) = 0;
                      *(_QWORD *)(v50 + 16) = v50 + 40;
                      *(_QWORD *)(v50 + 24) = 2;
                      *(_DWORD *)(v50 + 32) = 0;
                      *(_BYTE *)(v50 + 36) = 1;
                      goto LABEL_80;
                    }
                  }
                }
              }
            }
          }
        }
      }
LABEL_16:
      v1 = (_QWORD *)v1[1];
      if ( v108 + 3 == v1 )
        goto LABEL_17;
    }
    if ( *(_BYTE *)(v8 + 64) )
      goto LABEL_14;
LABEL_50:
    v26 = sub_B11F60(v8 + 80);
    if ( (unsigned int)((__int64)(*(_QWORD *)(v26 + 24) - *(_QWORD *)(v26 + 16)) >> 3) )
      goto LABEL_14;
    if ( !sub_B13320(v8) )
      goto LABEL_14;
    v27 = sub_B13320(v8);
    v28 = (_BYTE *)sub_BD3990(v27);
    v29 = v28;
    if ( *v28 != 60 )
      goto LABEL_14;
    if ( !(unsigned __int8)sub_B4D040(v28) )
      goto LABEL_14;
    sub_B4CED0(&v125, v29, v110);
    if ( v126[0] )
    {
      if ( v125.m128i_i8[8] )
        goto LABEL_14;
    }
    if ( v120 )
    {
      v30 = 1;
      v105 = ((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4);
      v31 = (v120 - 1) & v105;
      v32 = 0;
      v33 = &v118[7 * v31];
      v34 = (_BYTE *)*v33;
      if ( v29 == (_BYTE *)*v33 )
      {
LABEL_58:
        v35 = v33 + 1;
        if ( !*((_BYTE *)v33 + 36) )
          goto LABEL_59;
        goto LABEL_96;
      }
      while ( v34 != (_BYTE *)-4096LL )
      {
        if ( v34 == (_BYTE *)-8192LL && !v32 )
          v32 = v33;
        v54 = v30 + 1;
        v55 = (v120 - 1) & (v31 + v30);
        v31 = v55;
        v33 = &v118[7 * v55];
        v34 = (_BYTE *)*v33;
        if ( v29 == (_BYTE *)*v33 )
          goto LABEL_58;
        v30 = v54;
      }
      if ( v32 )
        v33 = v32;
      ++v117;
      v56 = v119 + 1;
      if ( 4 * ((int)v119 + 1) < 3 * v120 )
      {
        if ( v120 - HIDWORD(v119) - v56 <= v120 >> 3 )
        {
          sub_AEC1C0((__int64)&v117, v120);
          if ( !v120 )
            goto LABEL_241;
          v65 = 1;
          v66 = (v120 - 1) & v105;
          v56 = v119 + 1;
          v67 = 0;
          v33 = &v118[7 * v66];
          v68 = *v33;
          if ( v29 != (_BYTE *)*v33 )
          {
            while ( v68 != -4096 )
            {
              if ( v68 == -8192 && !v67 )
                v67 = v33;
              v102 = v65 + 1;
              v103 = (v120 - 1) & (v66 + v65);
              v66 = v103;
              v33 = &v118[7 * v103];
              v68 = *v33;
              if ( v29 == (_BYTE *)*v33 )
                goto LABEL_93;
              v65 = v102;
            }
            if ( v67 )
              v33 = v67;
          }
        }
LABEL_93:
        LODWORD(v119) = v56;
        if ( *v33 != -4096 )
          --HIDWORD(v119);
        *v33 = v29;
        v35 = v33 + 1;
        v33[1] = 0;
        v33[2] = v33 + 5;
        v33[3] = 2;
        *((_DWORD *)v33 + 8) = 0;
        *((_BYTE *)v33 + 36) = 1;
LABEL_96:
        v57 = (__int64 *)v35[1];
        v58 = *((unsigned int *)v35 + 5);
        v59 = &v57[v58];
        if ( v57 != v59 )
        {
          while ( *v57 != v8 )
          {
            if ( v59 == ++v57 )
              goto LABEL_99;
          }
LABEL_60:
          v36 = v124;
          if ( v124 )
            goto LABEL_61;
          goto LABEL_101;
        }
LABEL_99:
        if ( (unsigned int)v58 < *((_DWORD *)v35 + 4) )
        {
          *((_DWORD *)v35 + 5) = v58 + 1;
          *v59 = v8;
          ++*v35;
          v36 = v124;
          if ( v124 )
          {
LABEL_61:
            v37 = 1;
            v106 = ((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4);
            v38 = (v36 - 1) & v106;
            v39 = 0;
            v40 = (__int64 *)(v122 + 88LL * v38);
            v41 = (_BYTE *)*v40;
            if ( v29 == (_BYTE *)*v40 )
            {
LABEL_62:
              v107 = (__int64)(v40 + 1);
              goto LABEL_63;
            }
            while ( v41 != (_BYTE *)-4096LL )
            {
              if ( !v39 && v41 == (_BYTE *)-8192LL )
                v39 = v40;
              v38 = (v36 - 1) & (v38 + v37);
              v40 = (__int64 *)(v122 + 88LL * v38);
              v41 = (_BYTE *)*v40;
              if ( v29 == (_BYTE *)*v40 )
                goto LABEL_62;
              ++v37;
            }
            if ( v39 )
              v40 = v39;
            ++v121;
            v61 = v123 + 1;
            if ( 4 * ((int)v123 + 1) < 3 * v36 )
            {
              if ( v36 - HIDWORD(v123) - v61 <= v36 >> 3 )
              {
                sub_AEC410((__int64)&v121, v36);
                if ( !v124 )
                {
LABEL_243:
                  LODWORD(v123) = v123 + 1;
                  BUG();
                }
                v73 = 1;
                v74 = (v124 - 1) & v106;
                v61 = v123 + 1;
                v75 = 0;
                v40 = (__int64 *)(v122 + 88LL * v74);
                v76 = (_BYTE *)*v40;
                if ( v29 != (_BYTE *)*v40 )
                {
                  while ( v76 != (_BYTE *)-4096LL )
                  {
                    if ( v76 == (_BYTE *)-8192LL && !v75 )
                      v75 = v40;
                    v74 = (v124 - 1) & (v74 + v73);
                    v40 = (__int64 *)(v122 + 88LL * v74);
                    v76 = (_BYTE *)*v40;
                    if ( v29 == (_BYTE *)*v40 )
                      goto LABEL_119;
                    ++v73;
                  }
                  if ( v75 )
                    v40 = v75;
                }
              }
              goto LABEL_119;
            }
LABEL_102:
            sub_AEC410((__int64)&v121, 2 * v36);
            if ( !v124 )
              goto LABEL_243;
            v60 = (v124 - 1) & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
            v61 = v123 + 1;
            v40 = (__int64 *)(v122 + 88LL * v60);
            v62 = (_BYTE *)*v40;
            if ( v29 != (_BYTE *)*v40 )
            {
              v63 = 1;
              v64 = 0;
              while ( v62 != (_BYTE *)-4096LL )
              {
                if ( v62 == (_BYTE *)-8192LL && !v64 )
                  v64 = v40;
                v60 = (v124 - 1) & (v60 + v63);
                v40 = (__int64 *)(v122 + 88LL * v60);
                v62 = (_BYTE *)*v40;
                if ( v29 == (_BYTE *)*v40 )
                  goto LABEL_119;
                ++v63;
              }
              if ( v64 )
                v40 = v64;
            }
LABEL_119:
            LODWORD(v123) = v61;
            if ( *v40 != -4096 )
              --HIDWORD(v123);
            *v40 = (__int64)v29;
            v107 = (__int64)(v40 + 1);
            v40[5] = (__int64)(v40 + 7);
            v40[6] = 0x200000000LL;
            *(_OWORD *)(v40 + 1) = 0;
            *(_OWORD *)(v40 + 3) = 0;
            *(_OWORD *)(v40 + 7) = 0;
            *(_OWORD *)(v40 + 9) = 0;
LABEL_63:
            v125.m128i_i64[0] = sub_B12000(v8 + 72);
            sub_AE7AF0((__int64)&v112, v8);
            v125.m128i_i64[1] = sub_B10CD0(&v112);
            if ( v112 )
              sub_B91220(&v112);
            sub_AEFB90(v107, &v125);
LABEL_14:
            while ( 1 )
            {
              v8 = *(_QWORD *)(v8 + 8);
              if ( v8 == v7 )
                goto LABEL_15;
              if ( !*(_BYTE *)(v8 + 32) )
              {
                if ( v7 == v8 )
                  goto LABEL_15;
                if ( !*(_BYTE *)(v8 + 64) )
                  goto LABEL_50;
              }
            }
          }
LABEL_101:
          ++v121;
          goto LABEL_102;
        }
LABEL_59:
        sub_C8CC70(v35, v8);
        goto LABEL_60;
      }
    }
    else
    {
      ++v117;
    }
    sub_AEC1C0((__int64)&v117, 2 * v120);
    if ( !v120 )
    {
LABEL_241:
      LODWORD(v119) = v119 + 1;
      BUG();
    }
    v69 = (v120 - 1) & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
    v56 = v119 + 1;
    v33 = &v118[7 * v69];
    v70 = *v33;
    if ( v29 != (_BYTE *)*v33 )
    {
      v71 = 1;
      v72 = 0;
      while ( v70 != -4096 )
      {
        if ( !v72 && v70 == -8192 )
          v72 = v33;
        v69 = (v120 - 1) & (v69 + v71);
        v33 = &v118[7 * v69];
        v70 = *v33;
        if ( v29 == (_BYTE *)*v33 )
          goto LABEL_93;
        ++v71;
      }
      if ( v72 )
        v33 = v72;
    }
    goto LABEL_93;
  }
  v9 = (_QWORD *)(a1 + 72);
LABEL_19:
  v10 = (unsigned __int64)v109;
  sub_AE9DC0(v9, v109, (__int64)&v121, v110);
  if ( !(_DWORD)v115 )
    goto LABEL_20;
  v77 = v116;
  v78 = v114;
  v79 = &v114[7 * v116];
  if ( v114 == v79 )
    goto LABEL_20;
  while ( 1 )
  {
    v80 = *v78;
    v81 = v78;
    LOBYTE(v1) = *v78 == -8192 || *v78 == -4096;
    if ( !(_BYTE)v1 )
      break;
    v78 += 7;
    if ( v79 == v78 )
      goto LABEL_20;
  }
  if ( v78 == v79 )
  {
LABEL_20:
    LODWORD(v1) = 0;
  }
  else
  {
    while ( 1 )
    {
      if ( (*(_BYTE *)(v80 + 7) & 0x20) != 0 )
      {
        v10 = 38;
        v82 = sub_B91C10(v80, 38);
        if ( v82 )
          sub_AE94B0(v82);
      }
      v83 = (_QWORD *)v81[2];
      if ( *((_BYTE *)v81 + 36) )
        v84 = *((unsigned int *)v81 + 7);
      else
        v84 = *((unsigned int *)v81 + 6);
      v85 = &v83[v84];
      if ( v83 != v85 )
      {
        while ( 1 )
        {
          v86 = *v83;
          v87 = v83;
          if ( *v83 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v85 == ++v83 )
            goto LABEL_155;
        }
        if ( v85 != v83 )
        {
          do
          {
            sub_B43D60(v86, v10, v84, v77);
            v99 = v87 + 1;
            if ( v87 + 1 == v85 )
              break;
            while ( 1 )
            {
              v86 = *v99;
              v87 = v99;
              if ( *v99 < 0xFFFFFFFFFFFFFFFELL )
                break;
              if ( v85 == ++v99 )
                goto LABEL_192;
            }
          }
          while ( v85 != v99 );
LABEL_192:
          LODWORD(v1) = 1;
        }
      }
LABEL_155:
      v81 += 7;
      if ( v81 == v79 )
        break;
      while ( *v81 == -8192 || *v81 == -4096 )
      {
        v81 += 7;
        if ( v79 == v81 )
        {
          if ( !(_DWORD)v119 )
            goto LABEL_22;
          goto LABEL_160;
        }
      }
      if ( v79 == v81 )
        break;
      v80 = *v81;
    }
  }
  if ( (_DWORD)v119 )
  {
LABEL_160:
    v88 = v118;
    v89 = &v118[7 * v120];
    if ( v118 != v89 )
    {
      while ( 1 )
      {
        v90 = *v88;
        v91 = v88;
        if ( *v88 != -8192 && v90 != -4096 )
          break;
        v88 += 7;
        if ( v89 == v88 )
          goto LABEL_22;
      }
      if ( v89 != v88 )
      {
        if ( (*(_BYTE *)(v90 + 7) & 0x20) == 0 )
          goto LABEL_184;
LABEL_167:
        v10 = 38;
        v92 = sub_B91C10(v90, 38);
        if ( v92 )
        {
          v93 = *(_QWORD *)(v92 + 8);
          v10 = v93 & 0xFFFFFFFFFFFFFFF8LL;
          if ( (v93 & 4) == 0 )
            v10 = 0;
          sub_B967C0(&v125, v10);
          v94 = (_QWORD *)v91[2];
          if ( !*((_BYTE *)v91 + 36) )
            goto LABEL_185;
        }
        else
        {
LABEL_184:
          while ( 1 )
          {
            v125.m128i_i64[0] = (__int64)v126;
            v125.m128i_i64[1] = 0x600000000LL;
            v94 = (_QWORD *)v91[2];
            if ( *((_BYTE *)v91 + 36) )
              break;
LABEL_185:
            v95 = &v94[*((unsigned int *)v91 + 6)];
LABEL_172:
            if ( v94 != v95 )
            {
              while ( 1 )
              {
                v96 = *v94;
                v97 = v94;
                if ( *v94 < 0xFFFFFFFFFFFFFFFELL )
                  break;
                if ( v95 == ++v94 )
                  goto LABEL_175;
              }
              if ( v94 != v95 )
              {
                do
                {
                  sub_B14290(v96);
                  v100 = v97 + 1;
                  if ( v97 + 1 == v95 )
                    break;
                  while ( 1 )
                  {
                    v96 = *v100;
                    v97 = v100;
                    if ( *v100 < 0xFFFFFFFFFFFFFFFELL )
                      break;
                    if ( v95 == ++v100 )
                      goto LABEL_200;
                  }
                }
                while ( v100 != v95 );
LABEL_200:
                LODWORD(v1) = 1;
              }
            }
LABEL_175:
            if ( (_BYTE *)v125.m128i_i64[0] != v126 )
              _libc_free(v125.m128i_i64[0], v10);
            v91 += 7;
            if ( v91 == v89 )
              goto LABEL_22;
            while ( 1 )
            {
              v98 = *v91;
              if ( *v91 != -8192 && v98 != -4096 )
                break;
              v91 += 7;
              if ( v89 == v91 )
                goto LABEL_22;
            }
            if ( v91 == v89 )
              goto LABEL_22;
            v90 = *v91;
            if ( (*(_BYTE *)(v98 + 7) & 0x20) != 0 )
              goto LABEL_167;
          }
        }
        v95 = &v94[*((unsigned int *)v91 + 7)];
        goto LABEL_172;
      }
    }
  }
LABEL_22:
  v11 = v124;
  if ( v124 )
  {
    v12 = v122;
    v13 = v122 + 88LL * v124;
    do
    {
      if ( *(_QWORD *)v12 != -4096 && *(_QWORD *)v12 != -8192 )
      {
        v14 = *(_QWORD *)(v12 + 40);
        if ( v14 != v12 + 56 )
          _libc_free(v14, v10);
        v10 = 16LL * *(unsigned int *)(v12 + 32);
        sub_C7D6A0(*(_QWORD *)(v12 + 16), v10, 8);
      }
      v12 += 88;
    }
    while ( v13 != v12 );
    v11 = v124;
  }
  v15 = 88 * v11;
  sub_C7D6A0(v122, 88 * v11, 8);
  v16 = v120;
  if ( v120 )
  {
    v17 = v118;
    v18 = &v118[7 * v120];
    do
    {
      while ( *v17 == -8192 || *v17 == -4096 || *((_BYTE *)v17 + 36) )
      {
        v17 += 7;
        if ( v18 == v17 )
          goto LABEL_38;
      }
      v19 = v17[2];
      v17 += 7;
      _libc_free(v19, v15);
    }
    while ( v18 != v17 );
LABEL_38:
    v16 = v120;
  }
  v20 = 56 * v16;
  sub_C7D6A0(v118, 56 * v16, 8);
  v21 = v116;
  if ( v116 )
  {
    v22 = v114;
    v23 = &v114[7 * v116];
    do
    {
      while ( *v22 == -4096 || *v22 == -8192 || *((_BYTE *)v22 + 36) )
      {
        v22 += 7;
        if ( v23 == v22 )
          goto LABEL_46;
      }
      v24 = v22[2];
      v22 += 7;
      _libc_free(v24, v20);
    }
    while ( v23 != v22 );
LABEL_46:
    v21 = v116;
  }
  sub_C7D6A0(v114, 56 * v21, 8);
  return (unsigned int)v1;
}
