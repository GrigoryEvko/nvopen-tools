// Function: sub_3436580
// Address: 0x3436580
//
void __fastcall sub_3436580(__int64 a1, _QWORD *a2, __m128i a3)
{
  __int64 v5; // r12
  __int64 v6; // rax
  __int64 v7; // rbx
  __int64 v8; // r13
  unsigned int v9; // esi
  __int64 v10; // r8
  __int64 v11; // rdi
  __int64 *v12; // rdx
  __int64 v13; // rcx
  unsigned int v14; // esi
  __int64 v15; // rcx
  unsigned int v16; // edx
  __int64 v17; // rax
  _QWORD *v18; // r8
  int v19; // edx
  __int64 v20; // rax
  unsigned int v21; // edx
  __int64 v22; // rbx
  unsigned int v23; // r12d
  _QWORD *v24; // rax
  __int64 v25; // rax
  unsigned __int16 v26; // dx
  __int64 v27; // rax
  _QWORD *v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rsi
  __int64 v31; // r12
  unsigned __int8 *v32; // rax
  unsigned __int8 *v33; // rbx
  int v34; // edx
  int v35; // r12d
  _QWORD *v36; // rax
  __int64 *v37; // rax
  int v38; // ecx
  int v39; // ecx
  unsigned __int64 v40; // rax
  int v41; // edx
  __int64 v42; // rcx
  __int64 v43; // rdi
  unsigned __int64 v44; // rsi
  int v45; // r9d
  int v46; // r11d
  unsigned int i; // edx
  __int64 v48; // rax
  unsigned int v49; // edx
  __int64 v50; // r13
  unsigned int v51; // r12d
  __int64 v52; // rbx
  __int64 v53; // rdi
  unsigned int v54; // esi
  __int64 (__fastcall *v55)(__int64, __int64, unsigned int); // rax
  int v56; // edx
  unsigned __int16 v57; // ax
  const __m128i *v58; // rax
  __int64 v59; // rdx
  _QWORD *v60; // r13
  __int64 v61; // rbx
  __m128i v62; // xmm1
  __int64 v63; // rax
  int v64; // ecx
  unsigned __int64 v65; // rdx
  unsigned __int64 v66; // rax
  __int64 *v67; // rbx
  __int64 v68; // rax
  __int64 v69; // r13
  __int64 v70; // rax
  unsigned int v71; // eax
  __int64 *v72; // rbx
  __int64 v73; // rdx
  __int64 v74; // r13
  unsigned int v75; // r10d
  __m128i *v76; // rax
  _QWORD *v77; // rsi
  __m128i *v78; // rbx
  __int64 v79; // r8
  __int64 v80; // r9
  int v81; // edx
  int v82; // r13d
  __int64 v83; // rax
  __m128i **v84; // rax
  _QWORD *v85; // rax
  int v86; // ebx
  __int64 v87; // r13
  int v88; // eax
  const __m128i *v89; // rsi
  __m128i *v90; // rax
  __int64 v91; // r8
  __int64 v92; // rbx
  int v93; // edx
  int v94; // r12d
  _QWORD *v95; // rax
  int v96; // eax
  int v97; // r9d
  int v98; // eax
  int v99; // esi
  __int64 v100; // rdi
  unsigned int v101; // edx
  __int64 v102; // r8
  int v103; // r10d
  __int64 *v104; // r9
  int v105; // eax
  int v106; // edx
  __int64 *v107; // r8
  __int64 v108; // rdi
  int v109; // r9d
  unsigned int v110; // r11d
  __int64 v111; // rsi
  int v112; // r12d
  __int64 v113; // rbx
  _QWORD *v114; // rax
  __m128i v115; // [rsp+0h] [rbp-1B0h]
  _QWORD *v116; // [rsp+10h] [rbp-1A0h]
  __int64 v117; // [rsp+18h] [rbp-198h]
  unsigned int v118; // [rsp+20h] [rbp-190h]
  int v119; // [rsp+28h] [rbp-188h]
  const __m128i *v120; // [rsp+28h] [rbp-188h]
  __m128i v121; // [rsp+80h] [rbp-130h] BYREF
  __int64 v122; // [rsp+90h] [rbp-120h]
  __m128i v123; // [rsp+A0h] [rbp-110h] BYREF
  __int64 v124; // [rsp+B0h] [rbp-100h]
  _QWORD *v125; // [rsp+C0h] [rbp-F0h] BYREF
  __int64 v126; // [rsp+C8h] [rbp-E8h]
  _QWORD v127[8]; // [rsp+D0h] [rbp-E0h] BYREF
  char *v128; // [rsp+110h] [rbp-A0h]
  char v129; // [rsp+128h] [rbp-88h] BYREF
  char *v130; // [rsp+130h] [rbp-80h]
  char v131; // [rsp+140h] [rbp-70h] BYREF
  char *v132; // [rsp+150h] [rbp-60h]
  char v133; // [rsp+160h] [rbp-50h] BYREF

  v5 = sub_B5B6B0((__int64)a2);
  v6 = sub_B5B890((__int64)a2);
  v7 = *(_QWORD *)(a1 + 960);
  v8 = v6;
  v9 = *(_DWORD *)(v7 + 240);
  if ( !v9 )
  {
    ++*(_QWORD *)(v7 + 216);
    goto LABEL_75;
  }
  v10 = *(_QWORD *)(v7 + 224);
  v11 = (v9 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
  v12 = (__int64 *)(v10 + 40 * v11);
  v13 = *v12;
  if ( v5 != *v12 )
  {
    v119 = 1;
    v37 = 0;
    while ( v13 != -4096 )
    {
      if ( v13 == -8192 && !v37 )
        v37 = v12;
      LODWORD(v11) = (v9 - 1) & (v119 + v11);
      v12 = (__int64 *)(v10 + 40LL * (unsigned int)v11);
      v13 = *v12;
      if ( v5 == *v12 )
        goto LABEL_3;
      ++v119;
    }
    v38 = *(_DWORD *)(v7 + 232);
    if ( !v37 )
      v37 = v12;
    ++*(_QWORD *)(v7 + 216);
    v39 = v38 + 1;
    if ( 4 * v39 < 3 * v9 )
    {
      if ( v9 - *(_DWORD *)(v7 + 236) - v39 > v9 >> 3 )
      {
LABEL_24:
        *(_DWORD *)(v7 + 232) = v39;
        if ( *v37 != -4096 )
          --*(_DWORD *)(v7 + 236);
        *v37 = v5;
        v14 = 0;
        v15 = 0;
        v37[1] = 0;
        v37[2] = 0;
        v37[3] = 0;
        *((_DWORD *)v37 + 8) = 0;
        goto LABEL_27;
      }
      sub_3435C30(v7 + 216, v9);
      v105 = *(_DWORD *)(v7 + 240);
      if ( v105 )
      {
        v106 = v105 - 1;
        v107 = 0;
        v108 = *(_QWORD *)(v7 + 224);
        v109 = 1;
        v110 = (v105 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
        v39 = *(_DWORD *)(v7 + 232) + 1;
        v37 = (__int64 *)(v108 + 40LL * v110);
        v111 = *v37;
        if ( v5 != *v37 )
        {
          while ( v111 != -4096 )
          {
            if ( v111 == -8192 && !v107 )
              v107 = v37;
            v110 = v106 & (v109 + v110);
            v37 = (__int64 *)(v108 + 40LL * v110);
            v111 = *v37;
            if ( v5 == *v37 )
              goto LABEL_24;
            ++v109;
          }
          if ( v107 )
            v37 = v107;
        }
        goto LABEL_24;
      }
LABEL_111:
      ++*(_DWORD *)(v7 + 232);
      BUG();
    }
LABEL_75:
    sub_3435C30(v7 + 216, 2 * v9);
    v98 = *(_DWORD *)(v7 + 240);
    if ( v98 )
    {
      v99 = v98 - 1;
      v100 = *(_QWORD *)(v7 + 224);
      v101 = (v98 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
      v39 = *(_DWORD *)(v7 + 232) + 1;
      v37 = (__int64 *)(v100 + 40LL * v101);
      v102 = *v37;
      if ( v5 != *v37 )
      {
        v103 = 1;
        v104 = 0;
        while ( v102 != -4096 )
        {
          if ( !v104 && v102 == -8192 )
            v104 = v37;
          v101 = v99 & (v103 + v101);
          v37 = (__int64 *)(v100 + 40LL * v101);
          v102 = *v37;
          if ( v5 == *v37 )
            goto LABEL_24;
          ++v103;
        }
        if ( v104 )
          v37 = v104;
      }
      goto LABEL_24;
    }
    goto LABEL_111;
  }
LABEL_3:
  v14 = *((_DWORD *)v12 + 8);
  v15 = v12[2];
  if ( v14 )
  {
    v16 = (v14 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v17 = v15 + 16LL * v16;
    v18 = *(_QWORD **)v17;
    if ( a2 == *(_QWORD **)v17 )
    {
LABEL_5:
      v19 = *(_DWORD *)(v17 + 8);
      if ( v19 != 3 )
        goto LABEL_6;
      goto LABEL_28;
    }
    v96 = 1;
    while ( v18 != (_QWORD *)-4096LL )
    {
      v97 = v96 + 1;
      v16 = (v14 - 1) & (v96 + v16);
      v17 = v15 + 16LL * v16;
      v18 = *(_QWORD **)v17;
      if ( a2 == *(_QWORD **)v17 )
        goto LABEL_5;
      v96 = v97;
    }
  }
LABEL_27:
  v17 = v15 + 16LL * v14;
  v19 = *(_DWORD *)(v17 + 8);
  if ( v19 != 3 )
  {
LABEL_6:
    if ( v19 == 2 )
    {
      v86 = *(_DWORD *)(v17 + 12);
      v87 = a2[1];
      v123.m128i_i8[4] = 0;
      v88 = sub_2E79000(*(__int64 **)(*(_QWORD *)(a1 + 864) + 40LL));
      sub_336FEE0(
        (__int64)&v125,
        *(_QWORD *)(*(_QWORD *)(a1 + 864) + 64LL),
        *(_QWORD *)(*(_QWORD *)(a1 + 864) + 16LL),
        v88,
        v86,
        v87,
        v123.m128i_i64[0]);
      v89 = *(const __m128i **)(a1 + 864);
      v90 = *(__m128i **)a1;
      v123.m128i_i32[2] = *(_DWORD *)(a1 + 848);
      v123.m128i_i64[0] = 0;
      v121 = _mm_loadu_si128(v89 + 24);
      if ( v90 )
      {
        if ( &v123 != &v90[3] )
        {
          v91 = v90[3].m128i_i64[0];
          v123.m128i_i64[0] = v91;
          if ( v91 )
          {
            sub_B96E90((__int64)&v123, v91, 1);
            v89 = *(const __m128i **)(a1 + 864);
          }
        }
      }
      v92 = sub_3370E50((__int64)&v125, (__int64)v89, *(_QWORD *)(a1 + 960), (__int64)&v123, (__int64)&v121, 0, 0);
      v94 = v93;
      if ( v123.m128i_i64[0] )
        sub_B91220((__int64)&v123, v123.m128i_i64[0]);
      v123.m128i_i64[0] = (__int64)a2;
      v95 = sub_337DC20(a1 + 8, v123.m128i_i64);
      *v95 = v92;
      *((_DWORD *)v95 + 2) = v94;
      if ( v132 != &v133 )
        _libc_free((unsigned __int64)v132);
      if ( v130 != &v131 )
        _libc_free((unsigned __int64)v130);
      if ( v128 != &v129 )
        _libc_free((unsigned __int64)v128);
      if ( v125 != v127 )
        _libc_free((unsigned __int64)v125);
    }
    else if ( v19 == 1 )
    {
      v50 = *(_QWORD *)(a1 + 864);
      v51 = *(_DWORD *)(v17 + 12);
      v52 = *(_QWORD *)(v50 + 16);
      v53 = sub_2E79000(*(__int64 **)(v50 + 40));
      v54 = *(_DWORD *)(v53 + 4);
      v55 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v52 + 32LL);
      if ( v55 == sub_2D42F30 )
      {
        v56 = sub_AE2980(v53, v54)[1];
        v57 = 2;
        if ( v56 != 1 )
        {
          v57 = 3;
          if ( v56 != 2 )
          {
            v57 = 4;
            if ( v56 != 4 )
            {
              v57 = 5;
              if ( v56 != 8 )
              {
                v57 = 6;
                if ( v56 != 16 )
                {
                  v57 = 7;
                  if ( v56 != 32 )
                  {
                    v57 = 8;
                    if ( v56 != 64 )
                      v57 = 9 * (v56 == 128);
                  }
                }
              }
            }
          }
        }
      }
      else
      {
        v57 = v55(v52, v53, v54);
      }
      v116 = sub_33EDBD0((_QWORD *)v50, v51, v57, 0, 1);
      v58 = *(const __m128i **)(a1 + 864);
      v117 = v59;
      v60 = (_QWORD *)v58[2].m128i_i64[1];
      v61 = v60[6];
      v115 = _mm_loadu_si128(v58 + 24);
      sub_2EAC300((__int64)&v121, (__int64)v60, v51, 0);
      v62 = _mm_load_si128(&v121);
      v125 = 0;
      v126 = 0;
      v127[0] = 0;
      v127[1] = 0;
      v63 = *(_QWORD *)(v61 + 8) + 40LL * (*(_DWORD *)(v61 + 32) + v51);
      v64 = *(unsigned __int8 *)(v63 + 16);
      v65 = *(_QWORD *)(v63 + 8);
      v123 = v62;
      v124 = v122;
      if ( v65 > 0x3FFFFFFFFFFFFFFBLL )
        LODWORD(v65) = -2;
      v66 = sub_2E7BD70(v60, 1u, v65, v64, (int)&v125, 0, *(_OWORD *)&v123, v122, 1u, 0, 0);
      v67 = (__int64 *)a2[1];
      v120 = (const __m128i *)v66;
      v68 = *(_QWORD *)(a1 + 864);
      v69 = *(_QWORD *)(v68 + 16);
      v70 = sub_2E79000(*(__int64 **)(v68 + 40));
      v71 = sub_2D5BAE0(v69, v70, v67, 0);
      v72 = *(__int64 **)(a1 + 864);
      v125 = 0;
      v74 = v73;
      v75 = v71;
      v76 = *(__m128i **)a1;
      LODWORD(v126) = *(_DWORD *)(a1 + 848);
      if ( v76 )
      {
        if ( &v125 != (_QWORD **)&v76[3] )
        {
          v77 = (_QWORD *)v76[3].m128i_i64[0];
          v125 = v77;
          if ( v77 )
          {
            v118 = v75;
            sub_B96E90((__int64)&v125, (__int64)v77, 1);
            v75 = v118;
          }
        }
      }
      v78 = sub_33F1A60(v72, v75, v74, (__int64)&v125, v115.m128i_i64[0], v115.m128i_i64[1], (__int64)v116, v117, v120);
      v82 = v81;
      if ( v125 )
        sub_B91220((__int64)&v125, (__int64)v125);
      v83 = *(unsigned int *)(a1 + 136);
      if ( v83 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 140) )
      {
        sub_C8D5F0(a1 + 128, (const void *)(a1 + 144), v83 + 1, 0x10u, v79, v80);
        v83 = *(unsigned int *)(a1 + 136);
      }
      v84 = (__m128i **)(*(_QWORD *)(a1 + 128) + 16 * v83);
      *v84 = v78;
      v84[1] = (__m128i *)1;
      ++*(_DWORD *)(a1 + 136);
      v125 = a2;
      v85 = sub_337DC20(a1 + 8, (__int64 *)&v125);
      *v85 = v78;
      *((_DWORD *)v85 + 2) = v82;
    }
    else
    {
      v20 = sub_338B750(a1, v8);
      v22 = v20;
      v23 = v21;
      if ( *(_DWORD *)(v20 + 24) != 51 )
        goto LABEL_9;
      v25 = *(_QWORD *)(v20 + 48) + 16LL * v21;
      v26 = *(_WORD *)v25;
      v27 = *(_QWORD *)(v25 + 8);
      v121.m128i_i16[0] = v26;
      v121.m128i_i64[1] = v27;
      if ( v26 )
      {
        if ( v26 == 1 || (unsigned __int16)(v26 - 504) <= 7u )
          BUG();
        v29 = 16LL * (v26 - 1);
        v28 = *(_QWORD **)&byte_444C4A0[v29];
        LOBYTE(v29) = byte_444C4A0[v29 + 8];
      }
      else
      {
        v28 = (_QWORD *)sub_3007260((__int64)&v121);
        v125 = v28;
        v126 = v29;
      }
      v123.m128i_i64[0] = (__int64)v28;
      v123.m128i_i8[8] = v29;
      if ( (unsigned __int64)sub_CA1930(&v123) <= 0x40 )
      {
        v30 = *(_QWORD *)(v22 + 80);
        v31 = *(_QWORD *)(a1 + 864);
        v123.m128i_i64[0] = v30;
        if ( v30 )
          sub_B96E90((__int64)&v123, v30, 1);
        v123.m128i_i32[2] = *(_DWORD *)(v22 + 72);
        v32 = sub_3400BD0(v31, 4278124286LL, (__int64)&v123, 8, 0, 0, a3, 0);
        v121.m128i_i64[0] = (__int64)a2;
        v33 = v32;
        v35 = v34;
        v36 = sub_337DC20(a1 + 8, v121.m128i_i64);
        *v36 = v33;
        *((_DWORD *)v36 + 2) = v35;
        if ( v123.m128i_i64[0] )
          sub_B91220((__int64)&v123, v123.m128i_i64[0]);
      }
      else
      {
LABEL_9:
        v123.m128i_i64[0] = (__int64)a2;
        v24 = sub_337DC20(a1 + 8, v123.m128i_i64);
        *v24 = v22;
        *((_DWORD *)v24 + 2) = v23;
      }
    }
    return;
  }
LABEL_28:
  v40 = sub_338B750(a1, v8);
  v42 = *(unsigned int *)(a1 + 296);
  v43 = *(_QWORD *)(a1 + 280);
  v44 = v40;
  v45 = v41;
  if ( !(_DWORD)v42 )
    goto LABEL_89;
  v46 = 1;
  for ( i = (v42 - 1) & (v41 + ((v40 >> 9) ^ (v40 >> 4))); ; i = (v42 - 1) & v49 )
  {
    v48 = v43 + 32LL * i;
    if ( v44 == *(_QWORD *)v48 && *(_DWORD *)(v48 + 8) == v45 )
      break;
    if ( !*(_QWORD *)v48 && *(_DWORD *)(v48 + 8) == -1 )
      goto LABEL_89;
    v49 = v46 + i;
    ++v46;
  }
  if ( v48 == v43 + 32 * v42 )
  {
LABEL_89:
    v112 = 0;
    v113 = 0;
  }
  else
  {
    v113 = *(_QWORD *)(v48 + 16);
    v112 = *(_DWORD *)(v48 + 24);
  }
  v125 = a2;
  v114 = sub_337DC20(a1 + 8, (__int64 *)&v125);
  *v114 = v113;
  *((_DWORD *)v114 + 2) = v112;
}
