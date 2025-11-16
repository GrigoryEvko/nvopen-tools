// Function: sub_20920A0
// Address: 0x20920a0
//
void __fastcall sub_20920A0(
        __int64 a1,
        __int64 a2,
        __int64 *a3,
        __int64 a4,
        __int64 a5,
        __m128i a6,
        __m128i a7,
        __m128i a8)
{
  __int64 *v9; // r13
  __int64 v10; // r11
  unsigned __int64 v11; // rdi
  unsigned int v12; // eax
  __int64 v13; // rdx
  unsigned __int64 v14; // rbx
  unsigned int v15; // r12d
  unsigned int v16; // r8d
  unsigned __int64 v17; // rsi
  __int64 v18; // rcx
  int v19; // r9d
  __int64 v20; // r15
  __int64 v21; // r15
  unsigned int v22; // eax
  unsigned int v23; // esi
  unsigned int v24; // edi
  __int64 v25; // r12
  __int64 *v26; // r14
  unsigned int v27; // r13d
  __int64 v28; // rax
  __int64 *v29; // r10
  __int64 v30; // r15
  int v31; // r9d
  __int64 *v32; // r10
  __int64 v33; // rsi
  __int64 v34; // rax
  int v35; // r8d
  __int64 v36; // rax
  __int64 v37; // rax
  __m128 *v38; // rax
  int v39; // r9d
  __int64 v40; // r11
  int v41; // r8d
  __int64 v42; // rdx
  __int64 v43; // rax
  __int64 v44; // rax
  __m128i *v45; // rax
  __int64 v46; // r11
  __int64 v47; // rax
  __int64 v48; // rsi
  __int64 v49; // rax
  __int64 v50; // rax
  __m128i *v51; // r12
  __int64 v52; // rsi
  unsigned int v53; // r10d
  unsigned int v54; // r13d
  __int64 v55; // rax
  __int64 v56; // rdx
  __int64 v57; // rax
  unsigned __int32 v58; // eax
  const void **v59; // rsi
  bool v60; // al
  __int64 v61; // rdi
  __int64 v62; // rax
  unsigned __int32 v63; // eax
  bool v64; // al
  bool v65; // r15
  __int64 v66; // [rsp+0h] [rbp-E0h]
  __int64 v67; // [rsp+0h] [rbp-E0h]
  __int64 v68; // [rsp+8h] [rbp-D8h]
  __int64 v69; // [rsp+8h] [rbp-D8h]
  __int64 v70; // [rsp+8h] [rbp-D8h]
  __int64 *v71; // [rsp+8h] [rbp-D8h]
  __int64 v72; // [rsp+8h] [rbp-D8h]
  __int64 v73; // [rsp+10h] [rbp-D0h]
  __int64 *v74; // [rsp+10h] [rbp-D0h]
  __int64 *v75; // [rsp+10h] [rbp-D0h]
  __int64 v76; // [rsp+10h] [rbp-D0h]
  bool v77; // [rsp+10h] [rbp-D0h]
  __int64 v78; // [rsp+10h] [rbp-D0h]
  __int64 v79; // [rsp+10h] [rbp-D0h]
  __int64 v80; // [rsp+10h] [rbp-D0h]
  __int64 *v81; // [rsp+10h] [rbp-D0h]
  __int64 *v82; // [rsp+18h] [rbp-C8h]
  __int64 *v83; // [rsp+18h] [rbp-C8h]
  int v84; // [rsp+18h] [rbp-C8h]
  __int64 v85; // [rsp+18h] [rbp-C8h]
  __int64 *v86; // [rsp+18h] [rbp-C8h]
  __int64 *v87; // [rsp+18h] [rbp-C8h]
  unsigned int v88; // [rsp+18h] [rbp-C8h]
  unsigned int v89; // [rsp+18h] [rbp-C8h]
  __int64 v90; // [rsp+20h] [rbp-C0h]
  int v91; // [rsp+20h] [rbp-C0h]
  unsigned int v92; // [rsp+20h] [rbp-C0h]
  __int64 v93; // [rsp+20h] [rbp-C0h]
  __int64 v94; // [rsp+20h] [rbp-C0h]
  unsigned int v95; // [rsp+20h] [rbp-C0h]
  unsigned int v96; // [rsp+20h] [rbp-C0h]
  __int64 v99; // [rsp+30h] [rbp-B0h]
  unsigned int v101; // [rsp+40h] [rbp-A0h]
  __int64 v102; // [rsp+40h] [rbp-A0h]
  __int64 v103; // [rsp+48h] [rbp-98h]
  unsigned int v104; // [rsp+48h] [rbp-98h]
  unsigned int v105; // [rsp+48h] [rbp-98h]
  unsigned int v106; // [rsp+48h] [rbp-98h]
  __int64 v107; // [rsp+50h] [rbp-90h] BYREF
  unsigned int v108; // [rsp+58h] [rbp-88h]
  __m128i v109; // [rsp+60h] [rbp-80h] BYREF
  __m128i v110; // [rsp+70h] [rbp-70h] BYREF
  __m128i v111; // [rsp+80h] [rbp-60h] BYREF
  __int64 v112; // [rsp+90h] [rbp-50h]
  __int64 v113; // [rsp+98h] [rbp-48h] BYREF
  unsigned int v114; // [rsp+A0h] [rbp-40h]
  unsigned int v115; // [rsp+A8h] [rbp-38h]
  unsigned int v116; // [rsp+ACh] [rbp-34h]

  v9 = a3;
  v11 = a3[2];
  v12 = *((_DWORD *)a3 + 10) >> 1;
  v13 = a3[1];
  v10 = v13;
  v14 = v11;
  v15 = v12 + *(_DWORD *)(v13 + 32);
  v103 = v11;
  if ( v12 + (unsigned __int64)*(unsigned int *)(v13 + 32) > 0x80000000 )
    v15 = 0x80000000;
  v16 = *(_DWORD *)(v11 + 32) + v12;
  v17 = v13 + 40;
  if ( *(unsigned int *)(v11 + 32) + (unsigned __int64)v12 > 0x80000000 )
    v16 = 0x80000000;
  v18 = v13;
  v19 = 0;
  if ( v11 <= v17 )
  {
LABEL_84:
    v103 = v11;
    v13 = v10;
    goto LABEL_22;
  }
  do
  {
    while ( 1 )
    {
      if ( v15 >= v16 && (v15 != v16 || (v19 & 1) == 0) )
      {
        v20 = *(unsigned int *)(v14 - 8);
        if ( v20 + (unsigned __int64)v16 > 0x80000000 )
        {
          v14 -= 40LL;
          v16 = 0x80000000;
        }
        else
        {
          v16 += v20;
          v14 -= 40LL;
        }
        goto LABEL_10;
      }
      v21 = *(unsigned int *)(v18 + 72);
      if ( v21 + (unsigned __int64)v15 <= 0x80000000 )
        break;
      v18 = v17;
      v15 = 0x80000000;
LABEL_10:
      v17 = v18 + 40;
      ++v19;
      if ( v14 <= v18 + 40 )
        goto LABEL_14;
    }
    v18 = v17;
    v15 += v21;
    ++v19;
    v17 += 40LL;
  }
  while ( v14 > v17 );
LABEL_14:
  v22 = -858993459 * ((v18 - v13) >> 3) + 1;
  v24 = -858993459 * ((__int64)(v11 - v14) >> 3) + 1;
  v23 = v24;
  if ( v22 <= v24 )
    v24 = -858993459 * ((v18 - v13) >> 3) + 1;
  if ( v24 <= 2 )
  {
    v104 = v16;
    v101 = v15;
    v25 = v18;
    v90 = a1;
    v26 = v9;
    while ( 1 )
    {
      if ( v22 < v23 )
      {
        if ( v23 <= 3 || (v54 = sub_2054800(v14, v14, v26[2]), v54 < (unsigned int)sub_2054800(v14, v26[1], v25)) )
        {
LABEL_21:
          v9 = v26;
          v16 = v104;
          v18 = v25;
          a1 = v90;
          v15 = v101;
          v13 = v9[1];
          v103 = v9[2];
          break;
        }
        v25 += 40;
        v14 += 40LL;
      }
      else
      {
        if ( v22 <= 3 )
        {
          v9 = v26;
          v18 = v25;
          v16 = v104;
          v15 = v101;
          a1 = v90;
          v10 = v9[1];
          v11 = v9[2];
          goto LABEL_84;
        }
        v27 = sub_2054800(v25, v26[1], v25);
        if ( v27 < (unsigned int)sub_2054800(v25, v14, v26[2]) )
          goto LABEL_21;
        v25 -= 40;
        v14 -= 40LL;
      }
      v13 = v26[1];
      v22 = -858993459 * ((v25 - v13) >> 3) + 1;
      v23 = -858993459 * ((__int64)(v26[2] - v14) >> 3) + 1;
      v53 = v23;
      if ( v22 <= v23 )
        v53 = -858993459 * ((v25 - v13) >> 3) + 1;
      if ( v53 > 2 )
      {
        v16 = v104;
        v18 = v25;
        v9 = v26;
        v103 = v26[2];
        v15 = v101;
        a1 = v90;
        break;
      }
    }
  }
LABEL_22:
  v102 = *(_QWORD *)(v14 + 8);
  v28 = *v9;
  v29 = *(__int64 **)(*v9 + 8);
  if ( v18 == v13 && !*(_DWORD *)v18 && *(_QWORD *)(v18 + 8) == v9[3] )
  {
    v62 = *(_QWORD *)(v18 + 16);
    v108 = *(_DWORD *)(v62 + 32);
    if ( v108 > 0x40 )
    {
      v67 = v13;
      v72 = v18;
      v81 = v29;
      v89 = v16;
      sub_16A4FD0((__int64)&v107, (const void **)(v62 + 24));
      v13 = v67;
      v18 = v72;
      v29 = v81;
      v16 = v89;
    }
    else
    {
      v107 = *(_QWORD *)(v62 + 24);
    }
    v69 = v13;
    v78 = v18;
    v87 = v29;
    v96 = v16;
    sub_16A7490((__int64)&v107, 1);
    v63 = v108;
    v108 = 0;
    v16 = v96;
    v29 = v87;
    v109.m128i_i32[2] = v63;
    v18 = v78;
    v109.m128i_i64[0] = v107;
    v13 = v69;
    if ( v63 <= 0x40 )
    {
      if ( v107 == *(_QWORD *)(v102 + 24) )
        goto LABEL_82;
    }
    else
    {
      v66 = v69;
      v70 = v78;
      v79 = v107;
      v64 = sub_16A5220((__int64)&v109, (const void **)(v102 + 24));
      v16 = v96;
      v29 = v87;
      v18 = v70;
      v65 = v64;
      v13 = v66;
      if ( v79 )
      {
        j_j___libc_free_0_0(v79);
        v16 = v96;
        v29 = v87;
        v18 = v70;
        v13 = v66;
        if ( v108 > 0x40 )
        {
          if ( v107 )
          {
            j_j___libc_free_0_0(v107);
            v13 = v66;
            v18 = v70;
            v29 = v87;
            v16 = v96;
          }
        }
      }
      if ( v65 )
      {
LABEL_82:
        v30 = *(_QWORD *)(v18 + 24);
        goto LABEL_26;
      }
    }
    v28 = *v9;
  }
  v68 = v13;
  v73 = v18;
  v82 = v29;
  v91 = v16;
  v30 = (__int64)sub_1E0B6F0(*(_QWORD *)(*(_QWORD *)(a1 + 712) + 8LL), *(_QWORD *)(v28 + 40));
  sub_1DD8DC0(*(_QWORD *)(*(_QWORD *)(a1 + 712) + 8LL) + 320LL, v30);
  v32 = v82;
  v33 = *v82;
  v34 = *(_QWORD *)v30 & 7LL;
  *(_QWORD *)(v30 + 8) = v82;
  v35 = v91;
  v33 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v30 = v33 | v34;
  *(_QWORD *)(v33 + 8) = v30;
  *v82 = v30 | *v82 & 7;
  v36 = v9[3];
  v109.m128i_i64[1] = v68;
  v110.m128i_i64[1] = v36;
  v109.m128i_i64[0] = v30;
  v111.m128i_i64[0] = v102;
  LODWORD(v36) = *((_DWORD *)v9 + 10);
  v110.m128i_i64[0] = v73;
  v111.m128i_i32[2] = (unsigned int)v36 >> 1;
  v37 = *(unsigned int *)(a2 + 8);
  if ( (unsigned int)v37 >= *(_DWORD *)(a2 + 12) )
  {
    sub_16CD150(a2, (const void *)(a2 + 16), 0, 48, v91, v31);
    v32 = v82;
    v35 = v91;
    v37 = *(unsigned int *)(a2 + 8);
  }
  a6 = _mm_loadu_si128(&v109);
  v83 = v32;
  v38 = (__m128 *)(*(_QWORD *)a2 + 48 * v37);
  v92 = v35;
  *v38 = (__m128)a6;
  a7 = _mm_loadu_si128(&v110);
  v38[1] = (__m128)a7;
  a8 = _mm_loadu_si128(&v111);
  v38[2] = (__m128)a8;
  ++*(_DWORD *)(a2 + 8);
  sub_2090460(a1, a4, a6, a7, a8);
  v29 = v83;
  v16 = v92;
LABEL_26:
  if ( v14 == v103 && !*(_DWORD *)v14 )
  {
    v56 = v9[4];
    if ( v56 )
    {
      v57 = *(_QWORD *)(v14 + 16);
      v108 = *(_DWORD *)(v57 + 32);
      if ( v108 > 0x40 )
      {
        v71 = v29;
        v80 = v56;
        v88 = v16;
        sub_16A4FD0((__int64)&v107, (const void **)(v57 + 24));
        v29 = v71;
        v56 = v80;
        v16 = v88;
      }
      else
      {
        v107 = *(_QWORD *)(v57 + 24);
      }
      v75 = v29;
      v85 = v56;
      v95 = v16;
      sub_16A7490((__int64)&v107, 1);
      v58 = v108;
      v108 = 0;
      v16 = v95;
      v109.m128i_i32[2] = v58;
      v29 = v75;
      v109.m128i_i64[0] = v107;
      if ( v58 <= 0x40 )
      {
        if ( v107 == *(_QWORD *)(v85 + 24) )
          goto LABEL_71;
      }
      else
      {
        v59 = (const void **)(v85 + 24);
        v76 = v107;
        v86 = v29;
        v60 = sub_16A5220((__int64)&v109, v59);
        v16 = v95;
        v29 = v86;
        if ( v76 )
        {
          v61 = v76;
          v77 = v60;
          j_j___libc_free_0_0(v61);
          v16 = v95;
          v29 = v86;
          v60 = v77;
          if ( v108 > 0x40 )
          {
            if ( v107 )
            {
              j_j___libc_free_0_0(v107);
              v60 = v77;
              v29 = v86;
              v16 = v95;
            }
          }
        }
        if ( v60 )
        {
LABEL_71:
          v46 = *(_QWORD *)(v14 + 24);
          goto LABEL_30;
        }
      }
    }
  }
  v74 = v29;
  v84 = v16;
  v93 = (__int64)sub_1E0B6F0(*(_QWORD *)(*(_QWORD *)(a1 + 712) + 8LL), *(_QWORD *)(*v9 + 40));
  sub_1DD8DC0(*(_QWORD *)(*(_QWORD *)(a1 + 712) + 8LL) + 320LL, v93);
  v40 = v93;
  v41 = v84;
  v42 = *v74;
  v43 = *(_QWORD *)v93;
  *(_QWORD *)(v93 + 8) = v74;
  v42 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v93 = v42 | v43 & 7;
  *(_QWORD *)(v42 + 8) = v93;
  *v74 = v93 | *v74 & 7;
  v109.m128i_i64[0] = v93;
  v110.m128i_i64[0] = v103;
  v109.m128i_i64[1] = v14;
  v110.m128i_i64[1] = v102;
  v111.m128i_i64[0] = v9[4];
  v111.m128i_i32[2] = *((_DWORD *)v9 + 10) >> 1;
  v44 = *(unsigned int *)(a2 + 8);
  if ( (unsigned int)v44 >= *(_DWORD *)(a2 + 12) )
  {
    sub_16CD150(a2, (const void *)(a2 + 16), 0, 48, v84, v39);
    v44 = *(unsigned int *)(a2 + 8);
    v40 = v93;
    v41 = v84;
  }
  v94 = v40;
  v45 = (__m128i *)(*(_QWORD *)a2 + 48 * v44);
  v105 = v41;
  *v45 = _mm_loadu_si128(&v109);
  v45[1] = _mm_loadu_si128(&v110);
  v45[2] = _mm_loadu_si128(&v111);
  ++*(_DWORD *)(a2 + 8);
  sub_2090460(a1, a4, a6, a7, a8);
  v46 = v94;
  v16 = v105;
LABEL_30:
  v47 = *(_QWORD *)a1;
  v108 = *(_DWORD *)(a1 + 536);
  if ( !v47 || &v107 == (__int64 *)(v47 + 48) || (v48 = *(_QWORD *)(v47 + 48), (v107 = v48) == 0) )
  {
    v55 = *v9;
    v111.m128i_i64[0] = v30;
    v109.m128i_i32[0] = 20;
    v109.m128i_i64[1] = a4;
    v110.m128i_i64[0] = 0;
    v110.m128i_i64[1] = v102;
    v111.m128i_i64[1] = v46;
    v112 = v55;
    v113 = 0;
LABEL_59:
    v115 = v15;
    v116 = v16;
    v114 = v108;
    v50 = a5;
    if ( *v9 != a5 )
      goto LABEL_37;
    goto LABEL_60;
  }
  v99 = v46;
  v106 = v16;
  sub_1623A60((__int64)&v107, v48, 2);
  v49 = *v9;
  v109.m128i_i32[0] = 20;
  v111.m128i_i64[0] = v30;
  v109.m128i_i64[1] = a4;
  v111.m128i_i64[1] = v99;
  v16 = v106;
  v110.m128i_i64[0] = 0;
  v110.m128i_i64[1] = v102;
  v112 = v49;
  v113 = v107;
  if ( !v107 )
    goto LABEL_59;
  sub_1623A60((__int64)&v113, v107, 2);
  v115 = v15;
  v114 = v108;
  v116 = v106;
  if ( v107 )
    sub_161E7C0((__int64)&v107, v107);
  v50 = a5;
  if ( *v9 == a5 )
  {
LABEL_60:
    sub_2069F40((__int64 *)a1, (__int64)&v109, v50, a6, a7, a8);
    goto LABEL_43;
  }
LABEL_37:
  v51 = *(__m128i **)(a1 + 592);
  if ( v51 == *(__m128i **)(a1 + 600) )
  {
    sub_2055B00((__int64 *)(a1 + 584), *(_QWORD *)(a1 + 592), (__int64)&v109);
  }
  else
  {
    if ( v51 )
    {
      v51->m128i_i32[0] = v109.m128i_i32[0];
      v51->m128i_i64[1] = v109.m128i_i64[1];
      v51[1] = v110;
      v51[2] = v111;
      v51[3].m128i_i64[0] = v112;
      v52 = v113;
      v51[3].m128i_i64[1] = v113;
      if ( v52 )
        sub_1623A60((__int64)&v51[3].m128i_i64[1], v52, 2);
      v51[4].m128i_i32[0] = v114;
      v51[4].m128i_i32[2] = v115;
      v51[4].m128i_i32[3] = v116;
      v51 = *(__m128i **)(a1 + 592);
    }
    *(_QWORD *)(a1 + 592) = v51 + 5;
  }
LABEL_43:
  if ( v113 )
    sub_161E7C0((__int64)&v113, v113);
}
