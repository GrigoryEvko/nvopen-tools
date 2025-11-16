// Function: sub_104A960
// Address: 0x104a960
//
unsigned __int8 *__fastcall sub_104A960(__int64 a1, unsigned __int8 *a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int8 *v5; // r13
  __int64 v6; // rax
  _QWORD *v7; // r15
  char *v11; // rsi
  unsigned __int8 v12; // r8
  __int64 *v13; // r10
  __int64 v14; // r12
  __int64 v15; // rax
  __int64 v16; // r15
  __int64 v17; // rcx
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // rax
  unsigned __int64 v23; // rdx
  _QWORD *v25; // rax
  int v26; // r9d
  __int64 v27; // r9
  __int64 v28; // r8
  __int64 v29; // rsi
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // r12
  unsigned __int8 *v33; // r9
  __int64 *v34; // r13
  __int64 v35; // r15
  __int64 v36; // rsi
  __int64 v37; // rax
  __int64 v38; // r9
  __int64 v39; // rdx
  unsigned __int64 v40; // r8
  __int64 v41; // rcx
  __int64 v42; // rdx
  __int64 v43; // rax
  unsigned __int8 v44; // al
  __int64 v45; // rax
  __int64 v46; // r8
  __int64 v47; // r9
  __int64 *v48; // rdi
  _BYTE **v49; // r13
  __int64 v50; // rbx
  _BYTE *v51; // rdi
  __int64 v52; // rax
  __int64 v53; // rax
  unsigned __int8 *v54; // r15
  unsigned __int8 *v55; // r9
  __int64 v56; // r8
  unsigned __int8 *v57; // r12
  __int64 v58; // r9
  _BYTE *v59; // rbx
  __int64 v60; // rax
  unsigned __int8 *v61; // rsi
  char v62; // r15
  unsigned __int8 *v63; // rax
  unsigned __int8 *v64; // r11
  char v65; // r10
  unsigned __int8 *v66; // r9
  unsigned __int8 *v67; // rsi
  unsigned __int8 *v68; // r15
  char *v69; // rax
  __int64 v70; // rdx
  char *v71; // rcx
  __int64 v72; // rdx
  char *v73; // rdx
  __int64 v74; // rdx
  __int64 v75; // rax
  unsigned __int8 *v76; // rax
  unsigned __int8 *v77; // r9
  __int64 v78; // r14
  __int64 v79; // rdx
  __int64 v80; // rbx
  int v81; // eax
  __int64 v82; // rcx
  char v83; // al
  __int64 v84; // rcx
  __int64 v85; // rax
  __int64 v86; // r8
  unsigned __int8 *v87; // rdx
  __int64 v88; // r13
  unsigned __int8 *v89; // r14
  unsigned __int8 *v90; // rax
  __int64 v91; // rsi
  char v92; // al
  __int64 v93; // rax
  signed __int64 v94; // rdx
  __int64 v95; // [rsp+8h] [rbp-118h]
  __int64 v96; // [rsp+18h] [rbp-108h]
  __int64 v97; // [rsp+18h] [rbp-108h]
  __int64 v98; // [rsp+20h] [rbp-100h]
  __int64 v99; // [rsp+20h] [rbp-100h]
  char v100; // [rsp+28h] [rbp-F8h]
  bool v101; // [rsp+28h] [rbp-F8h]
  unsigned __int8 *v102; // [rsp+28h] [rbp-F8h]
  __int64 v103; // [rsp+28h] [rbp-F8h]
  __int64 v104; // [rsp+30h] [rbp-F0h]
  unsigned __int8 *v105; // [rsp+30h] [rbp-F0h]
  unsigned __int8 *v106; // [rsp+30h] [rbp-F0h]
  unsigned __int8 *v107; // [rsp+30h] [rbp-F0h]
  __int64 v108; // [rsp+30h] [rbp-F0h]
  unsigned __int8 *v109; // [rsp+30h] [rbp-F0h]
  unsigned __int8 *v111; // [rsp+48h] [rbp-D8h] BYREF
  __int64 v112[8]; // [rsp+50h] [rbp-D0h] BYREF
  __int16 v113; // [rsp+90h] [rbp-90h]
  __m128i v114; // [rsp+A0h] [rbp-80h] BYREF
  __int64 v115; // [rsp+B0h] [rbp-70h] BYREF
  __int64 v116; // [rsp+B8h] [rbp-68h]
  __int64 v117; // [rsp+C0h] [rbp-60h]
  __int64 v118; // [rsp+C8h] [rbp-58h]
  __int64 v119; // [rsp+D0h] [rbp-50h]
  __int64 v120; // [rsp+D8h] [rbp-48h]
  __int16 v121; // [rsp+E0h] [rbp-40h]

  v5 = a2;
  if ( *a2 <= 0x1Cu )
    return v5;
  v6 = *(unsigned int *)(a1 + 40);
  v7 = *(_QWORD **)(a1 + 32);
  v111 = a2;
  v11 = (char *)&v7[v6];
  if ( v11 != (char *)sub_104A710(v7, (__int64)v11, (__int64 *)&v111) )
  {
    if ( a3 != *((_QWORD *)v5 + 5) )
      return v5;
    v104 = a1 + 32;
    v25 = sub_104A710(v7, (__int64)v11, v13);
    if ( v11 != (char *)(v25 + 1) )
    {
      memmove(v25, v25 + 1, v11 - (char *)(v25 + 1));
      v5 = v111;
      v26 = *(_DWORD *)(a1 + 40);
    }
    v27 = (unsigned int)(v26 - 1);
    *(_DWORD *)(a1 + 40) = v27;
    v28 = *v5;
    if ( (_BYTE)v28 == 84 )
    {
      v29 = *((_QWORD *)v5 - 1);
      v30 = 0x1FFFFFFFE0LL;
      if ( (*((_DWORD *)v5 + 1) & 0x7FFFFFF) != 0 )
      {
        v31 = 0;
        do
        {
          if ( a4 == *(_QWORD *)(v29 + 32LL * *((unsigned int *)v5 + 18) + 8 * v31) )
          {
            v30 = 32 * v31;
            goto LABEL_23;
          }
          ++v31;
        }
        while ( (*((_DWORD *)v5 + 1) & 0x7FFFFFF) != (_DWORD)v31 );
        v30 = 0x1FFFFFFFE0LL;
      }
LABEL_23:
      v16 = *(_QWORD *)(v29 + v30);
      if ( *(_BYTE *)v16 > 0x1Cu )
      {
        if ( v27 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
        {
          sub_C8D5F0(v104, (const void *)(a1 + 48), v27 + 1, 8u, v28, v27);
          v27 = *(unsigned int *)(a1 + 40);
        }
        *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8 * v27) = v16;
        ++*(_DWORD *)(a1 + 40);
      }
      return (unsigned __int8 *)v16;
    }
    if ( !sub_104A7D0((__int64)v5) )
      return 0;
    v53 = 32LL * (*((_DWORD *)v5 + 1) & 0x7FFFFFF);
    if ( (v5[7] & 0x40) != 0 )
    {
      v54 = (unsigned __int8 *)*((_QWORD *)v5 - 1);
      v55 = &v54[v53];
    }
    else
    {
      v55 = v5;
      v54 = &v5[-v53];
    }
    if ( v55 != v54 )
    {
      v56 = a3;
      v57 = v55;
      v58 = a4;
      do
      {
        v59 = *(_BYTE **)v54;
        if ( **(_BYTE **)v54 > 0x1Cu )
        {
          v60 = *(unsigned int *)(a1 + 40);
          if ( v60 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
          {
            v97 = v58;
            v99 = v56;
            sub_C8D5F0(v104, (const void *)(a1 + 48), v60 + 1, 8u, v56, v58);
            v60 = *(unsigned int *)(a1 + 40);
            v58 = v97;
            v56 = v99;
          }
          *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8 * v60) = v59;
          ++*(_DWORD *)(a1 + 40);
        }
        v54 += 32;
      }
      while ( v57 != v54 );
      a3 = v56;
      a4 = v58;
      v12 = *v5;
    }
  }
  if ( (unsigned int)v12 - 67 > 0xC )
  {
    if ( v12 == 63 )
    {
      v114.m128i_i64[0] = (__int64)&v115;
      v114.m128i_i64[1] = 0x800000000LL;
      if ( (v5[7] & 0x40) != 0 )
      {
        v33 = (unsigned __int8 *)*((_QWORD *)v5 - 1);
        v105 = &v33[32 * (*((_DWORD *)v5 + 1) & 0x7FFFFFF)];
      }
      else
      {
        v105 = v5;
        v33 = &v5[-32 * (*((_DWORD *)v5 + 1) & 0x7FFFFFF)];
      }
      if ( v33 != v105 )
      {
        v100 = 0;
        v96 = (__int64)v5;
        v34 = (__int64 *)v33;
        do
        {
          v35 = *v34;
          v36 = *v34;
          v37 = sub_104A960(a1, *v34, a3, a4, a5);
          if ( !v37 )
          {
            v48 = (__int64 *)v114.m128i_i64[0];
            v16 = 0;
            goto LABEL_56;
          }
          v100 |= v37 != v35;
          v39 = v114.m128i_u32[2];
          v40 = v114.m128i_u32[2] + 1LL;
          if ( v40 > v114.m128i_u32[3] )
          {
            v36 = (__int64)&v115;
            v95 = v37;
            sub_C8D5F0((__int64)&v114, &v115, v114.m128i_u32[2] + 1LL, 8u, v40, v38);
            v39 = v114.m128i_u32[2];
            v37 = v95;
          }
          v34 += 4;
          *(_QWORD *)(v114.m128i_i64[0] + 8 * v39) = v37;
          ++v114.m128i_i32[2];
        }
        while ( v105 != (unsigned __int8 *)v34 );
        if ( v100 )
        {
          v41 = *(_QWORD *)(a1 + 8);
          v42 = *(_QWORD *)(a1 + 16);
          v112[2] = 0;
          memset(&v112[5], 0, 24);
          v43 = *(_QWORD *)(a1 + 24);
          v112[0] = v41;
          v112[1] = v42;
          v112[3] = a5;
          v113 = 257;
          v112[4] = v43;
          v44 = sub_B4DE20(v96);
          v36 = *(_QWORD *)v114.m128i_i64[0];
          v45 = sub_100E380(
                  *(_QWORD *)(v96 + 72),
                  *(_QWORD *)v114.m128i_i64[0],
                  (_QWORD *)(v114.m128i_i64[0] + 8),
                  v114.m128i_u32[2] - 1LL,
                  v44,
                  v112);
          v48 = (__int64 *)v114.m128i_i64[0];
          v16 = v45;
          if ( v45 )
          {
            v49 = (_BYTE **)v114.m128i_i64[0];
            v50 = v114.m128i_i64[0] + 8LL * v114.m128i_u32[2];
            if ( v50 != v114.m128i_i64[0] )
            {
              do
              {
                v51 = *v49;
                v36 = a1 + 32;
                ++v49;
                sub_104A830(v51, a1 + 32);
              }
              while ( (_BYTE **)v50 != v49 );
            }
            if ( *(_BYTE *)v16 > 0x1Cu )
            {
              v52 = *(unsigned int *)(a1 + 40);
              if ( v52 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
              {
                v36 = a1 + 48;
                sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), v52 + 1, 8u, v46, v47);
                v52 = *(unsigned int *)(a1 + 40);
              }
              *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8 * v52) = v16;
              ++*(_DWORD *)(a1 + 40);
            }
LABEL_55:
            v48 = (__int64 *)v114.m128i_i64[0];
          }
          else
          {
            v78 = *(_QWORD *)(*(_QWORD *)v114.m128i_i64[0] + 16LL);
            if ( v78 )
            {
              v79 = a4;
              while ( 1 )
              {
                v80 = *(_QWORD *)(v78 + 24);
                if ( *(_BYTE *)v80 == 63
                  && *(_QWORD *)(v96 + 8) == *(_QWORD *)(v80 + 8)
                  && *(_QWORD *)(v96 + 72) == *(_QWORD *)(v80 + 72) )
                {
                  v81 = *(_DWORD *)(v80 + 4);
                  v82 = v114.m128i_u32[2];
                  v36 = v81 & 0x7FFFFFF;
                  if ( v36 == v114.m128i_u32[2] )
                  {
                    v36 = *(_QWORD *)(v80 + 40);
                    if ( *(_QWORD *)(a3 + 72) == *(_QWORD *)(v36 + 72) )
                    {
                      if ( !a5 )
                        goto LABEL_105;
                      v108 = v79;
                      v83 = sub_B19720(a5, v36, v79);
                      v79 = v108;
                      if ( v83 )
                        break;
                    }
                  }
                }
LABEL_96:
                v78 = *(_QWORD *)(v78 + 8);
                if ( !v78 )
                  goto LABEL_55;
              }
              v82 = v114.m128i_u32[2];
              v81 = *(_DWORD *)(v80 + 4);
LABEL_105:
              v48 = (__int64 *)v114.m128i_i64[0];
              v36 = v80 - 32LL * (v81 & 0x7FFFFFF);
              v84 = v114.m128i_i64[0] + 8 * v82;
              v85 = 0;
              while ( v84 != v114.m128i_i64[0] + v85 )
              {
                v86 = *(_QWORD *)(v114.m128i_i64[0] + v85);
                v85 += 8;
                if ( v86 != *(_QWORD *)(v36 + 4 * v85 - 32) )
                  goto LABEL_96;
              }
              v16 = v80;
            }
          }
        }
        else
        {
          v48 = (__int64 *)v114.m128i_i64[0];
          v16 = v96;
        }
LABEL_56:
        if ( v48 != &v115 )
          _libc_free(v48, v36);
        return (unsigned __int8 *)v16;
      }
      return v5;
    }
    if ( v12 != 42 )
      return 0;
    v61 = (v5[7] & 0x40) != 0 ? (unsigned __int8 *)*((_QWORD *)v5 - 1) : &v5[-32 * (*((_DWORD *)v5 + 1) & 0x7FFFFFF)];
    if ( **((_BYTE **)v61 + 4) != 17 )
      return 0;
    v98 = *((_QWORD *)v61 + 4);
    v62 = sub_B44900((__int64)v5);
    v101 = sub_B448F0((__int64)v5);
    v63 = (unsigned __int8 *)sub_104A960(a1, *(_QWORD *)v61, a3, a4, a5);
    v64 = v63;
    if ( !v63 )
      return 0;
    v65 = v101;
    v66 = (unsigned __int8 *)v98;
    if ( *v63 != 42 )
      goto LABEL_85;
    v67 = (unsigned __int8 *)*((_QWORD *)v63 - 4);
    if ( *v67 != 17 )
      goto LABEL_85;
    v106 = v63;
    v68 = (unsigned __int8 *)*((_QWORD *)v63 - 8);
    v66 = (unsigned __int8 *)sub_AD57C0(v98, v67, 0, 0);
    v69 = *(char **)(a1 + 32);
    v70 = 8LL * *(unsigned int *)(a1 + 40);
    v71 = &v69[v70];
    v72 = v70 >> 5;
    if ( v72 )
    {
      v73 = &v69[32 * v72];
      while ( v106 != *(unsigned __int8 **)v69 )
      {
        if ( v106 == *((unsigned __int8 **)v69 + 1) )
        {
          v69 += 8;
          break;
        }
        if ( v106 == *((unsigned __int8 **)v69 + 2) )
        {
          v69 += 16;
          break;
        }
        if ( v106 == *((unsigned __int8 **)v69 + 3) )
        {
          v69 += 24;
          break;
        }
        v69 += 32;
        if ( v73 == v69 )
          goto LABEL_131;
      }
LABEL_83:
      if ( v71 != v69 )
      {
        v103 = (__int64)v66;
        sub_104A830(v106, a1 + 32);
        v66 = (unsigned __int8 *)v103;
        if ( *v68 > 0x1Cu )
        {
          v93 = *(unsigned int *)(a1 + 40);
          if ( v93 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
          {
            sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), v93 + 1, 8u, a1 + 32, v103);
            v93 = *(unsigned int *)(a1 + 40);
            v66 = (unsigned __int8 *)v103;
          }
          v64 = v68;
          v65 = 0;
          *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8 * v93) = v68;
          v62 = 0;
          ++*(_DWORD *)(a1 + 40);
          goto LABEL_85;
        }
      }
LABEL_84:
      v64 = v68;
      v65 = 0;
      v62 = 0;
LABEL_85:
      v74 = *(_QWORD *)(a1 + 16);
      v75 = *(_QWORD *)(a1 + 24);
      v102 = v66;
      v114.m128i_i64[0] = *(_QWORD *)(a1 + 8);
      v114.m128i_i64[1] = v74;
      v116 = a5;
      v117 = v75;
      v107 = v64;
      v115 = 0;
      v118 = 0;
      v119 = 0;
      v120 = 0;
      v121 = 257;
      v76 = sub_101BE10(v64, v66, v62, v65, &v114);
      v77 = v102;
      v16 = (__int64)v76;
      if ( v76 )
      {
        sub_104A830(v107, a1 + 32);
        if ( *(_BYTE *)v16 <= 0x1Cu )
          return (unsigned __int8 *)v16;
        v22 = *(unsigned int *)(a1 + 40);
        v23 = v22 + 1;
        if ( v22 + 1 <= (unsigned __int64)*(unsigned int *)(a1 + 44) )
          goto LABEL_11;
        goto LABEL_10;
      }
      if ( (v5[7] & 0x40) != 0 )
        v87 = (unsigned __int8 *)*((_QWORD *)v5 - 1);
      else
        v87 = &v5[-32 * (*((_DWORD *)v5 + 1) & 0x7FFFFFF)];
      if ( v107 == *(unsigned __int8 **)v87 && v102 == *((unsigned __int8 **)v87 + 4) )
        return v5;
      v88 = *((_QWORD *)v107 + 2);
      if ( v88 )
      {
        v89 = v107;
        while ( 1 )
        {
          v16 = *(_QWORD *)(v88 + 24);
          if ( *(_BYTE *)v16 == 42 )
          {
            v90 = *(unsigned __int8 **)(v16 - 64);
            if ( v90 )
            {
              if ( v89 == v90 && v77 == *(unsigned __int8 **)(v16 - 32) )
              {
                v91 = *(_QWORD *)(v16 + 40);
                if ( *(_QWORD *)(a3 + 72) == *(_QWORD *)(v91 + 72) )
                {
                  v109 = v77;
                  if ( !a5 )
                    break;
                  v92 = sub_B19720(a5, v91, a4);
                  v77 = v109;
                  if ( v92 )
                    break;
                }
              }
            }
          }
          v88 = *(_QWORD *)(v88 + 8);
          if ( !v88 )
            return 0;
        }
        return (unsigned __int8 *)v16;
      }
      return 0;
    }
LABEL_131:
    v94 = v71 - v69;
    if ( v71 - v69 != 16 )
    {
      if ( v94 != 24 )
      {
        if ( v94 != 8 )
          goto LABEL_84;
        goto LABEL_134;
      }
      if ( v106 == *(unsigned __int8 **)v69 )
        goto LABEL_83;
      v69 += 8;
    }
    if ( v106 == *(unsigned __int8 **)v69 )
      goto LABEL_83;
    v69 += 8;
LABEL_134:
    if ( v106 != *(unsigned __int8 **)v69 )
      goto LABEL_84;
    goto LABEL_83;
  }
  v14 = sub_104A960(a1, *((_QWORD *)v5 - 4), a3, a4, a5);
  if ( !v14 )
    return 0;
  v15 = *((_QWORD *)v5 - 4);
  if ( v15 == v14 )
  {
    v16 = (__int64)v5;
    if ( v15 )
      return (unsigned __int8 *)v16;
  }
  v17 = *(_QWORD *)(a1 + 8);
  v18 = *(_QWORD *)(a1 + 16);
  v115 = 0;
  v19 = *(_QWORD *)(a1 + 24);
  v118 = 0;
  v114.m128i_i64[0] = v17;
  v121 = 257;
  v114.m128i_i64[1] = v18;
  v116 = a5;
  v117 = v19;
  v119 = 0;
  v120 = 0;
  v16 = sub_1002A60((unsigned int)*v5 - 29, (unsigned __int8 *)v14, *((_QWORD *)v5 + 1), v114.m128i_i64);
  if ( !v16 )
  {
    v32 = *(_QWORD *)(v14 + 16);
    if ( v32 )
    {
      while ( 1 )
      {
        v16 = *(_QWORD *)(v32 + 24);
        if ( (unsigned __int8)(*(_BYTE *)v16 - 67) <= 0xCu
          && *(_BYTE *)v16 == *v5
          && *(_QWORD *)(v16 + 8) == *((_QWORD *)v5 + 1)
          && (!a5 || (unsigned __int8)sub_B19720(a5, *(_QWORD *)(v16 + 40), a4)) )
        {
          break;
        }
        v32 = *(_QWORD *)(v32 + 8);
        if ( !v32 )
          return 0;
      }
      return (unsigned __int8 *)v16;
    }
    return 0;
  }
  sub_104A830((_BYTE *)v14, a1 + 32);
  if ( *(_BYTE *)v16 > 0x1Cu )
  {
    v22 = *(unsigned int *)(a1 + 40);
    v23 = v22 + 1;
    if ( v22 + 1 <= (unsigned __int64)*(unsigned int *)(a1 + 44) )
    {
LABEL_11:
      *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8 * v22) = v16;
      ++*(_DWORD *)(a1 + 40);
      return (unsigned __int8 *)v16;
    }
LABEL_10:
    sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), v23, 8u, v20, v21);
    v22 = *(unsigned int *)(a1 + 40);
    goto LABEL_11;
  }
  return (unsigned __int8 *)v16;
}
