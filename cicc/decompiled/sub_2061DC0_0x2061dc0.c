// Function: sub_2061DC0
// Address: 0x2061dc0
//
const __m128i *__fastcall sub_2061DC0(
        const __m128i *a1,
        __int64 a2,
        __m128i *a3,
        __int64 a4,
        __m128i a5,
        __m128i a6,
        __m128i a7)
{
  __int64 v9; // rdx
  __int64 v10; // r15
  __int64 v11; // rsi
  __int64 v12; // rax
  int v13; // r14d
  __int64 v14; // r12
  __int64 v15; // r15
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  int v19; // r9d
  __int64 *v20; // rax
  __int64 v21; // rdx
  __int64 v22; // r10
  __int64 v23; // r11
  __int64 v24; // rax
  __int64 v25; // rsi
  _QWORD *v26; // rax
  __int64 v27; // rdx
  __int64 *v28; // rax
  __m128i *v29; // rcx
  __int64 v30; // rdx
  __int64 v31; // r14
  __int64 v32; // rax
  __int64 v33; // r15
  __int64 *v34; // rax
  __int64 v35; // rdx
  __int64 v36; // r10
  __int64 v37; // r11
  __int64 v38; // rax
  __int64 v39; // rsi
  _QWORD *v40; // rax
  __int64 v41; // rdx
  __int64 v42; // rax
  int v43; // eax
  _QWORD *v44; // rax
  _QWORD *v45; // r8
  int v46; // r9d
  unsigned int v48; // esi
  _QWORD *v49; // r8
  __int64 v50; // rdi
  unsigned int v51; // r15d
  unsigned int v52; // ecx
  __int64 *v53; // rax
  __int64 v54; // rdx
  _QWORD *v55; // rax
  __int64 v56; // r10
  __int64 v57; // rdx
  int v58; // r9d
  __int64 v59; // rdi
  unsigned __int32 v60; // eax
  __int64 v61; // r15
  __int64 v62; // rcx
  __int64 v63; // rax
  _DWORD *v64; // rax
  __int64 v65; // rax
  __int64 v66; // r11
  int v67; // eax
  int v68; // eax
  __int64 v69; // rax
  __int32 v70; // r10d
  __int64 *v71; // r11
  _QWORD *v72; // rcx
  __int32 v73; // ecx
  __int32 v74; // edx
  _QWORD *v75; // rcx
  int v76; // esi
  int v77; // esi
  __int64 v78; // r8
  __int64 v79; // rcx
  __int64 v80; // rdi
  int v81; // r11d
  __int64 *v82; // r9
  int v83; // r9d
  int v84; // r9d
  __int64 v85; // rdi
  __int64 v86; // rcx
  unsigned int v87; // edx
  __int32 v88; // r8d
  __int64 v89; // r11
  int v90; // ecx
  int v91; // ecx
  __int64 v92; // rdi
  __int64 *v93; // r8
  __int64 v94; // r15
  int v95; // r10d
  __int64 v96; // rsi
  int v97; // r9d
  int v98; // r9d
  __int64 v99; // rdi
  __int32 v100; // r8d
  __int64 v101; // rcx
  unsigned int v102; // edx
  __m128i v103; // [rsp+0h] [rbp-E0h] BYREF
  __int64 v104; // [rsp+10h] [rbp-D0h]
  __int64 v105; // [rsp+18h] [rbp-C8h]
  _QWORD *v106; // [rsp+20h] [rbp-C0h]
  __m128i *v107; // [rsp+28h] [rbp-B8h]
  __int64 v108; // [rsp+30h] [rbp-B0h]
  __int64 v109; // [rsp+38h] [rbp-A8h]
  __int64 v110; // [rsp+40h] [rbp-A0h]
  __int64 v111; // [rsp+48h] [rbp-98h]
  __int64 v112; // [rsp+50h] [rbp-90h]
  __int64 v113; // [rsp+58h] [rbp-88h]
  __m128i v114; // [rsp+60h] [rbp-80h]
  __int64 *v115; // [rsp+70h] [rbp-70h]
  __int64 v116; // [rsp+78h] [rbp-68h]
  __int64 v117; // [rsp+80h] [rbp-60h]
  __int64 v118; // [rsp+88h] [rbp-58h]
  __int64 v119; // [rsp+98h] [rbp-48h] BYREF
  __int64 v120; // [rsp+A0h] [rbp-40h] BYREF
  int v121; // [rsp+A8h] [rbp-38h]

  v107 = a3;
  v9 = *(_QWORD *)(a2 + 552);
  v119 = a4;
  v10 = *(_QWORD *)(*(_QWORD *)(v9 + 32) + 32LL);
  v106 = *(_QWORD **)(v9 + 32);
  v105 = v10;
  if ( !a4 )
  {
    v14 = 0;
    goto LABEL_12;
  }
  v11 = 1;
  v12 = sub_38BFA60(v10 + 168, 1);
  v13 = *(_DWORD *)(v10 + 1728);
  v14 = v12;
  if ( v13 )
  {
    v48 = *((_DWORD *)v106 + 122);
    v49 = v106 + 58;
    if ( v48 )
    {
      v50 = v106[59];
      v51 = ((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4);
      v52 = (v48 - 1) & v51;
      v53 = (__int64 *)(v50 + 16LL * v52);
      v54 = *v53;
      if ( v14 == *v53 )
        goto LABEL_34;
      v70 = 1;
      v71 = 0;
      while ( v54 != -8 )
      {
        if ( v54 == -16 && !v71 )
          v71 = v53;
        v52 = (v48 - 1) & (v70 + v52);
        v103.m128i_i32[0] = v70 + 1;
        v53 = (__int64 *)(v50 + 16LL * v52);
        v54 = *v53;
        if ( v14 == *v53 )
          goto LABEL_34;
        v70 = v103.m128i_i32[0];
      }
      v72 = v106;
      if ( v71 )
        v53 = v71;
      ++v106[58];
      v73 = *((_DWORD *)v72 + 120);
      v74 = v73 + 1;
      v103.m128i_i32[0] = v73;
      if ( 4 * (v73 + 1) < 3 * v48 )
      {
        if ( v48 - *((_DWORD *)v106 + 121) - v74 > v48 >> 3 )
        {
LABEL_55:
          v75 = v106;
          *((_DWORD *)v106 + 120) = v74;
          if ( *v53 != -8 )
            --*((_DWORD *)v75 + 121);
          *v53 = v14;
          *((_DWORD *)v53 + 2) = 0;
LABEL_34:
          *((_DWORD *)v53 + 2) = v13;
          v55 = sub_1FE1990(*(_QWORD *)(a2 + 712) + 48LL, &v119);
          v11 = *(unsigned int *)(a2 + 752);
          v56 = (__int64)v55;
          if ( (_DWORD)v11 )
          {
            v57 = v55[1];
            v58 = v11 - 1;
            v59 = *(_QWORD *)(a2 + 736);
            v60 = (v11 - 1) & (((unsigned int)v57 >> 9) ^ ((unsigned int)v57 >> 4));
            v61 = v59 + 40LL * v60;
            v62 = *(_QWORD *)v61;
            if ( v57 == *(_QWORD *)v61 )
            {
LABEL_36:
              v63 = *(unsigned int *)(v61 + 16);
              if ( (unsigned int)v63 >= *(_DWORD *)(v61 + 20) )
              {
                v11 = v61 + 24;
                sub_16CD150(v61 + 8, (const void *)(v61 + 24), 0, 4, a2 + 728, v58);
                v64 = (_DWORD *)(*(_QWORD *)(v61 + 8) + 4LL * *(unsigned int *)(v61 + 16));
              }
              else
              {
                v64 = (_DWORD *)(*(_QWORD *)(v61 + 8) + 4 * v63);
              }
LABEL_38:
              *v64 = v13;
              v65 = v105;
              ++*(_DWORD *)(v61 + 16);
              *(_DWORD *)(v65 + 1728) = 0;
              goto LABEL_3;
            }
            v103.m128i_i32[0] = 1;
            v66 = 0;
            v104 = v57;
            while ( v62 != -8 )
            {
              if ( !v66 && v62 == -16 )
                v66 = v61;
              v60 = v58 & (v103.m128i_i32[0] + v60);
              v61 = v59 + 40LL * v60;
              v62 = *(_QWORD *)v61;
              if ( v104 == *(_QWORD *)v61 )
                goto LABEL_36;
              ++v103.m128i_i32[0];
            }
            v67 = *(_DWORD *)(a2 + 744);
            if ( v66 )
              v61 = v66;
            ++*(_QWORD *)(a2 + 728);
            v68 = v67 + 1;
            if ( 4 * v68 < (unsigned int)(3 * v11) )
            {
              if ( (int)v11 - *(_DWORD *)(a2 + 748) - v68 > (unsigned int)v11 >> 3 )
              {
LABEL_46:
                *(_DWORD *)(a2 + 744) = v68;
                if ( *(_QWORD *)v61 != -8 )
                  --*(_DWORD *)(a2 + 748);
                v69 = *(_QWORD *)(v56 + 8);
                *(_QWORD *)(v61 + 16) = 0x400000000LL;
                *(_QWORD *)v61 = v69;
                v64 = (_DWORD *)(v61 + 24);
                *(_QWORD *)(v61 + 8) = v61 + 24;
                goto LABEL_38;
              }
              v103.m128i_i64[0] = v56;
              sub_1D52430(a2 + 728, v11);
              v97 = *(_DWORD *)(a2 + 752);
              if ( v97 )
              {
                v56 = v103.m128i_i64[0];
                v98 = v97 - 1;
                v99 = *(_QWORD *)(a2 + 736);
                v89 = 0;
                v100 = 1;
                v101 = *(_QWORD *)(v103.m128i_i64[0] + 8);
                v102 = v98 & (((unsigned int)v101 >> 9) ^ ((unsigned int)v101 >> 4));
                v61 = v99 + 40LL * v102;
                v11 = *(_QWORD *)v61;
                v68 = *(_DWORD *)(a2 + 744) + 1;
                if ( *(_QWORD *)v61 == v101 )
                  goto LABEL_46;
                while ( v11 != -8 )
                {
                  if ( !v89 && v11 == -16 )
                    v89 = v61;
                  v102 = v98 & (v100 + v102);
                  v103.m128i_i32[0] = v100 + 1;
                  v61 = v99 + 40LL * v102;
                  v11 = *(_QWORD *)v61;
                  if ( v101 == *(_QWORD *)v61 )
                    goto LABEL_46;
                  v100 = v103.m128i_i32[0];
                }
                goto LABEL_71;
              }
              goto LABEL_114;
            }
          }
          else
          {
            ++*(_QWORD *)(a2 + 728);
          }
          v103.m128i_i64[0] = v56;
          sub_1D52430(a2 + 728, 2 * v11);
          v83 = *(_DWORD *)(a2 + 752);
          if ( v83 )
          {
            v56 = v103.m128i_i64[0];
            v84 = v83 - 1;
            v85 = *(_QWORD *)(a2 + 736);
            v86 = *(_QWORD *)(v103.m128i_i64[0] + 8);
            v87 = v84 & (((unsigned int)v86 >> 9) ^ ((unsigned int)v86 >> 4));
            v61 = v85 + 40LL * v87;
            v11 = *(_QWORD *)v61;
            v68 = *(_DWORD *)(a2 + 744) + 1;
            if ( v86 == *(_QWORD *)v61 )
              goto LABEL_46;
            v88 = 1;
            v89 = 0;
            while ( v11 != -8 )
            {
              if ( !v89 && v11 == -16 )
                v89 = v61;
              v87 = v84 & (v88 + v87);
              v103.m128i_i32[0] = v88 + 1;
              v61 = v85 + 40LL * v87;
              v11 = *(_QWORD *)v61;
              if ( v86 == *(_QWORD *)v61 )
                goto LABEL_46;
              v88 = v103.m128i_i32[0];
            }
LABEL_71:
            if ( v89 )
              v61 = v89;
            goto LABEL_46;
          }
LABEL_114:
          ++*(_DWORD *)(a2 + 744);
          BUG();
        }
        sub_205F230((__int64)v49, v48);
        v90 = *((_DWORD *)v106 + 122);
        if ( v90 )
        {
          v91 = v90 - 1;
          v92 = v106[59];
          v93 = 0;
          LODWORD(v94) = v91 & v51;
          v95 = 1;
          v103.m128i_i32[0] = *((_DWORD *)v106 + 120);
          v74 = v103.m128i_i32[0] + 1;
          v53 = (__int64 *)(v92 + 16LL * (unsigned int)v94);
          v96 = *v53;
          if ( v14 != *v53 )
          {
            while ( v96 != -8 )
            {
              if ( v96 == -16 && !v93 )
                v93 = v53;
              v94 = v91 & (unsigned int)(v94 + v95);
              v53 = (__int64 *)(v92 + 16 * v94);
              v96 = *v53;
              if ( v14 == *v53 )
                goto LABEL_55;
              ++v95;
            }
            if ( v93 )
              v53 = v93;
          }
          goto LABEL_55;
        }
LABEL_113:
        ++*((_DWORD *)v106 + 120);
        BUG();
      }
    }
    else
    {
      ++v106[58];
    }
    sub_205F230((__int64)v49, 2 * v48);
    v76 = *((_DWORD *)v106 + 122);
    if ( v76 )
    {
      v77 = v76 - 1;
      v78 = v106[59];
      LODWORD(v79) = v77 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
      v103.m128i_i32[0] = *((_DWORD *)v106 + 120);
      v74 = v103.m128i_i32[0] + 1;
      v53 = (__int64 *)(v78 + 16LL * (unsigned int)v79);
      v80 = *v53;
      if ( v14 != *v53 )
      {
        v81 = 1;
        v82 = 0;
        while ( v80 != -8 )
        {
          if ( !v82 && v80 == -16 )
            v82 = v53;
          v79 = v77 & (unsigned int)(v79 + v81);
          v53 = (__int64 *)(v78 + 16 * v79);
          v80 = *v53;
          if ( v14 == *v53 )
            goto LABEL_55;
          ++v81;
        }
        if ( v82 )
          v53 = v82;
      }
      goto LABEL_55;
    }
    goto LABEL_113;
  }
LABEL_3:
  sub_2051C20((__int64 *)a2, *(double *)a5.m128i_i64, *(double *)a6.m128i_i64, a7);
  v15 = *(_QWORD *)(a2 + 552);
  v20 = sub_2051DF0((__int64 *)a2, *(double *)a5.m128i_i64, *(double *)a6.m128i_i64, a7, v11, v16, v17, v18, v19);
  v120 = 0;
  v22 = (__int64)v20;
  v23 = v21;
  v24 = *(_QWORD *)a2;
  v121 = *(_DWORD *)(a2 + 536);
  if ( v24 )
  {
    if ( &v120 != (__int64 *)(v24 + 48) )
    {
      v25 = *(_QWORD *)(v24 + 48);
      v120 = v25;
      if ( v25 )
      {
        v103.m128i_i64[0] = v22;
        v103.m128i_i64[1] = v21;
        sub_1623A60((__int64)&v120, v25, 2);
        v23 = v103.m128i_i64[1];
        v22 = v103.m128i_i64[0];
      }
    }
  }
  v26 = sub_1D2AAE0((_QWORD *)v15, (__int64)&v120, v22, v23, v14);
  if ( v26 )
  {
    v104 = v27;
    v103.m128i_i64[0] = (__int64)v26;
    nullsub_686();
    v117 = v103.m128i_i64[0];
    v118 = v104;
    *(_QWORD *)(v15 + 176) = v103.m128i_i64[0];
    *(_DWORD *)(v15 + 184) = v118;
    sub_1D23870();
  }
  else
  {
    v109 = v27;
    v108 = 0;
    *(_QWORD *)(v15 + 176) = 0;
    *(_DWORD *)(v15 + 184) = v109;
  }
  if ( v120 )
    sub_161E7C0((__int64)&v120, v120);
  v28 = sub_2051C20((__int64 *)a2, *(double *)a5.m128i_i64, *(double *)a6.m128i_i64, a7);
  v29 = v107;
  v116 = v30;
  v115 = v28;
  v107->m128i_i64[0] = (__int64)v28;
  v29->m128i_i32[2] = v116;
  v9 = *(_QWORD *)(a2 + 552);
LABEL_12:
  sub_2056920((__int64)a1, *(__m128i **)(v9 + 16), v107, a5, a6, a7);
  if ( a1[1].m128i_i64[0] )
  {
    v31 = *(_QWORD *)(a2 + 552);
    v103 = _mm_loadu_si128(a1 + 1);
    nullsub_686();
    a5 = _mm_load_si128(&v103);
    v114 = a5;
    *(_QWORD *)(v31 + 176) = a5.m128i_i64[0];
    *(_DWORD *)(v31 + 184) = v114.m128i_i32[2];
    sub_1D23870();
  }
  else
  {
    *(_BYTE *)(a2 + 760) = 1;
    *(_DWORD *)(a2 + 400) = 0;
  }
  if ( v119 )
  {
    v32 = sub_38BFA60(v105 + 168, 1);
    v33 = *(_QWORD *)(a2 + 552);
    v105 = v32;
    v34 = sub_2051C20((__int64 *)a2, *(double *)a5.m128i_i64, *(double *)a6.m128i_i64, a7);
    v120 = 0;
    v36 = (__int64)v34;
    v37 = v35;
    v38 = *(_QWORD *)a2;
    v121 = *(_DWORD *)(a2 + 536);
    if ( v38 )
    {
      if ( &v120 != (__int64 *)(v38 + 48) )
      {
        v39 = *(_QWORD *)(v38 + 48);
        v120 = v39;
        if ( v39 )
        {
          v103.m128i_i64[0] = v36;
          v103.m128i_i64[1] = v35;
          sub_1623A60((__int64)&v120, v39, 2);
          v37 = v103.m128i_i64[1];
          v36 = v103.m128i_i64[0];
        }
      }
    }
    v40 = sub_1D2AAE0((_QWORD *)v33, (__int64)&v120, v36, v37, v105);
    if ( v40 )
    {
      v104 = v41;
      v103.m128i_i64[0] = (__int64)v40;
      nullsub_686();
      v112 = v103.m128i_i64[0];
      v113 = v104;
      *(_QWORD *)(v33 + 176) = v103.m128i_i64[0];
      *(_DWORD *)(v33 + 184) = v113;
      sub_1D23870();
    }
    else
    {
      v111 = v41;
      v110 = 0;
      *(_QWORD *)(v33 + 176) = 0;
      *(_DWORD *)(v33 + 184) = v111;
    }
    if ( v120 )
      sub_161E7C0((__int64)&v120, v120);
    v42 = sub_15E38F0(**(_QWORD **)(a2 + 712));
    v43 = sub_14DD7D0(v42);
    if ( *((_BYTE *)v106 + 523) && (unsigned int)(v43 - 7) <= 3 )
    {
      sub_1F658C0(
        *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 552) + 32LL) + 88LL),
        v107[6].m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL,
        v14,
        v105);
    }
    else
    {
      v44 = sub_1FE1990(*(_QWORD *)(a2 + 712) + 48LL, &v119);
      sub_1E0CC60(v106, v44[1], v14, v105, v45, v46);
    }
  }
  return a1;
}
