// Function: sub_2B6ACB0
// Address: 0x2b6acb0
//
__int64 __fastcall sub_2B6ACB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rsi
  __int64 v9; // rcx
  __int64 v10; // rdi
  __int64 v11; // r13
  __int64 v12; // rax
  __m128i v13; // xmm1
  __int64 *v14; // rbx
  __m128i v15; // xmm0
  __int64 v16; // rax
  __int64 v17; // r13
  _BYTE *v18; // r12
  unsigned int *v19; // r14
  __int64 v20; // rdi
  int v21; // eax
  unsigned int v22; // edx
  __int64 v23; // rax
  __int64 result; // rax
  __int64 v25; // rdi
  char v26; // al
  unsigned int v27; // r15d
  unsigned int v28; // edx
  _BYTE *v29; // r12
  __int64 *v30; // r15
  __int64 v31; // rdi
  int v32; // edx
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rsi
  __int64 v37; // rbx
  __int64 v38; // rsi
  unsigned __int64 v39; // rax
  __int64 v40; // rdi
  char v41; // al
  unsigned int v42; // r10d
  unsigned int v43; // edx
  _BYTE *v44; // r12
  __int64 v45; // rdi
  int v46; // edx
  __int64 v47; // rax
  __int64 v48; // rax
  unsigned __int64 v49; // rax
  __int64 v50; // rdi
  char v51; // al
  unsigned int v52; // r10d
  unsigned int v53; // edx
  __int64 v54; // rdx
  __int64 v55; // rdx
  __int64 v56; // rdx
  __int64 v57; // r8
  __int64 v58; // r9
  __int64 v59; // r9
  unsigned __int64 v60; // rax
  unsigned __int64 v61; // rax
  unsigned __int64 v62; // rax
  int v63; // r8d
  int v64; // r9d
  int v65; // r9d
  int v66; // r10d
  int v67; // r10d
  int v68; // r10d
  char *v69; // rbx
  __int64 v70; // r12
  __int64 **v71; // rax
  int v72; // r12d
  _QWORD **v73; // r13
  __int64 v74; // rax
  unsigned int v75; // ebx
  __int64 v76; // rax
  __int64 v77; // rax
  int v78; // r12d
  __int64 v79; // rbx
  __int64 v80; // rax
  unsigned int v81; // [rsp+Ch] [rbp-144h]
  unsigned int v82; // [rsp+Ch] [rbp-144h]
  __int64 *v84; // [rsp+18h] [rbp-138h]
  __int64 *v85; // [rsp+28h] [rbp-128h]
  unsigned __int64 v86; // [rsp+30h] [rbp-120h] BYREF
  unsigned int v87; // [rsp+38h] [rbp-118h]
  __m128i v88; // [rsp+40h] [rbp-110h] BYREF
  __int64 v89; // [rsp+50h] [rbp-100h]
  __int64 v90; // [rsp+58h] [rbp-F8h]
  __m128i v91; // [rsp+60h] [rbp-F0h]
  __m128i v92[3]; // [rsp+70h] [rbp-E0h] BYREF
  __m128i v93; // [rsp+A0h] [rbp-B0h] BYREF
  unsigned int *v94; // [rsp+B0h] [rbp-A0h]
  unsigned int *v95; // [rsp+B8h] [rbp-98h]
  __m128i v96; // [rsp+D0h] [rbp-80h] BYREF
  __int64 v97; // [rsp+E0h] [rbp-70h]
  __int64 v98; // [rsp+E8h] [rbp-68h]
  __int64 v99; // [rsp+F0h] [rbp-60h] BYREF
  __int64 v100; // [rsp+F8h] [rbp-58h]
  __int64 v101; // [rsp+100h] [rbp-50h]
  __int64 v102; // [rsp+108h] [rbp-48h]
  __int16 v103; // [rsp+110h] [rbp-40h]

  v6 = *(__int64 **)(a1 + 24);
  v7 = *(_QWORD *)(a1 + 32);
  v8 = *(_QWORD *)(a1 + 16);
  v9 = v6[2];
  v10 = v6[1];
  v92[0].m128i_i64[1] = v7;
  v11 = *(unsigned int *)(v8 + 8);
  v12 = *v6;
  v90 = v7;
  v92[0].m128i_i64[0] = v9;
  v13 = _mm_loadu_si128(v92);
  v14 = *(__int64 **)v8;
  v88.m128i_i64[0] = v12;
  v11 *= 8;
  v88.m128i_i64[1] = v10;
  v92[2] = v13;
  v15 = _mm_loadu_si128(&v88);
  v84 = (__int64 *)((char *)v14 + v11);
  v89 = v9;
  v97 = v9;
  v96 = v15;
  v91 = v15;
  v92[1] = v15;
  v98 = v13.m128i_i64[1];
  v95 = (unsigned int *)v13.m128i_i64[1];
  v16 = v11 >> 3;
  v17 = v11 >> 5;
  v94 = (unsigned int *)v9;
  v93 = v15;
  if ( !v17 )
  {
LABEL_53:
    if ( v16 != 2 )
    {
      if ( v16 != 3 )
      {
        if ( v16 != 1 )
          goto LABEL_27;
LABEL_56:
        if ( sub_2B1EE10(v93.m128i_i64, (_BYTE *)*v14, v95) )
          goto LABEL_27;
        goto LABEL_11;
      }
      if ( !sub_2B1EE10(v93.m128i_i64, (_BYTE *)*v14, v95) )
      {
LABEL_11:
        result = 0;
        if ( v84 == v14 )
          goto LABEL_27;
        return result;
      }
      ++v14;
    }
    if ( sub_2B1EE10(v93.m128i_i64, (_BYTE *)*v14, v95) )
    {
      ++v14;
      goto LABEL_56;
    }
    goto LABEL_11;
  }
  v85 = &v14[4 * v17];
  while ( 1 )
  {
    v18 = (_BYTE *)*v14;
    v19 = v95;
    if ( *(_BYTE *)*v14 == 13 )
      goto LABEL_18;
    v9 = *(_BYTE *)(v93.m128i_i64[0] + 88) & 1;
    if ( (*(_BYTE *)(v93.m128i_i64[0] + 88) & 1) != 0 )
    {
      v20 = v93.m128i_i64[0] + 96;
      v21 = 3;
    }
    else
    {
      v54 = *(unsigned int *)(v93.m128i_i64[0] + 104);
      v20 = *(_QWORD *)(v93.m128i_i64[0] + 96);
      if ( !(_DWORD)v54 )
        goto LABEL_71;
      v21 = v54 - 1;
    }
    v22 = v21 & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
    a5 = v20 + 72LL * v22;
    a6 = *(_QWORD *)a5;
    if ( v18 == *(_BYTE **)a5 )
      goto LABEL_7;
    v63 = 1;
    while ( a6 != -4096 )
    {
      v66 = v63 + 1;
      v22 = v21 & (v63 + v22);
      a5 = v20 + 72LL * v22;
      a6 = *(_QWORD *)a5;
      if ( v18 == *(_BYTE **)a5 )
        goto LABEL_7;
      v63 = v66;
    }
    if ( (_BYTE)v9 )
    {
      v57 = 288;
      goto LABEL_72;
    }
    v54 = *(unsigned int *)(v93.m128i_i64[0] + 104);
LABEL_71:
    v57 = 72 * v54;
LABEL_72:
    a5 = v20 + v57;
LABEL_7:
    v23 = 288;
    if ( !(_BYTE)v9 )
      v23 = 72LL * *(unsigned int *)(v93.m128i_i64[0] + 104);
    if ( a5 != v20 + v23 && *(_DWORD *)(a5 + 16) > 1u )
      goto LABEL_11;
    v25 = *v14;
    v96 = (__m128i)*(unsigned __int64 *)(v93.m128i_i64[0] + 3344);
    v97 = 0;
    v98 = 0;
    v99 = 0;
    v100 = 0;
    v101 = 0;
    v102 = 0;
    v103 = 257;
    v26 = sub_9AC470(v25, &v96, 0);
    if ( *(_BYTE *)v93.m128i_i64[1] && v26 )
      goto LABEL_16;
    v27 = *v19;
    v28 = *v94;
    if ( *v94 <= *v19 )
      goto LABEL_16;
    v87 = *v94;
    if ( v28 > 0x40 )
    {
      sub_C43690((__int64)&v86, 0, 0);
      v28 = v87;
      if ( v27 == v87 )
        goto LABEL_84;
    }
    else
    {
      v86 = 0;
    }
    if ( v27 > 0x3F || v28 > 0x40 )
      sub_C43C90(&v86, v27, v28);
    else
      v86 |= 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v27 + 64 - (unsigned __int8)v28) << v27;
LABEL_84:
    v60 = *(_QWORD *)(v93.m128i_i64[0] + 3344);
    v103 = 257;
    v96 = (__m128i)v60;
    v97 = 0;
    v98 = 0;
    v99 = 0;
    v100 = 0;
    v101 = 0;
    v102 = 0;
    if ( !(unsigned __int8)sub_9AC230((__int64)v18, (__int64)&v86, &v96, 0) )
    {
      if ( v87 > 0x40 && v86 )
        j_j___libc_free_0_0(v86);
LABEL_16:
      if ( !sub_2B17B70(&v93, v18, v19) )
        goto LABEL_11;
      goto LABEL_17;
    }
    if ( v87 > 0x40 && v86 )
      j_j___libc_free_0_0(v86);
LABEL_17:
    v19 = v95;
LABEL_18:
    v29 = (_BYTE *)v14[1];
    v30 = v14 + 1;
    if ( *v29 == 13 )
      goto LABEL_38;
    a5 = *(_BYTE *)(v93.m128i_i64[0] + 88) & 1;
    if ( (*(_BYTE *)(v93.m128i_i64[0] + 88) & 1) != 0 )
    {
      v31 = v93.m128i_i64[0] + 96;
      v32 = 3;
    }
    else
    {
      v55 = *(unsigned int *)(v93.m128i_i64[0] + 104);
      v31 = *(_QWORD *)(v93.m128i_i64[0] + 96);
      if ( !(_DWORD)v55 )
        goto LABEL_74;
      v32 = v55 - 1;
    }
    v9 = v32 & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
    a6 = v31 + 72 * v9;
    v33 = *(_QWORD *)a6;
    if ( v29 == *(_BYTE **)a6 )
      goto LABEL_22;
    v64 = 1;
    while ( v33 != -4096 )
    {
      v67 = v64 + 1;
      v9 = v32 & (unsigned int)(v64 + v9);
      a6 = v31 + 72LL * (unsigned int)v9;
      v33 = *(_QWORD *)a6;
      if ( v29 == *(_BYTE **)a6 )
        goto LABEL_22;
      v64 = v67;
    }
    if ( (_BYTE)a5 )
    {
      v58 = 288;
      goto LABEL_75;
    }
    v55 = *(unsigned int *)(v93.m128i_i64[0] + 104);
LABEL_74:
    v58 = 72 * v55;
LABEL_75:
    a6 = v31 + v58;
LABEL_22:
    v34 = 288;
    if ( !(_BYTE)a5 )
      v34 = 72LL * *(unsigned int *)(v93.m128i_i64[0] + 104);
    if ( a6 != v31 + v34 && *(_DWORD *)(a6 + 16) > 1u )
      break;
    v39 = *(_QWORD *)(v93.m128i_i64[0] + 3344);
    v40 = v14[1];
    v103 = 257;
    v96 = (__m128i)v39;
    v97 = 0;
    v98 = 0;
    v99 = 0;
    v100 = 0;
    v101 = 0;
    v102 = 0;
    v41 = sub_9AC470(v40, &v96, 0);
    if ( *(_BYTE *)v93.m128i_i64[1] && v41 )
      goto LABEL_36;
    v42 = *v19;
    v43 = *v94;
    if ( *v94 <= *v19 )
      goto LABEL_36;
    v87 = *v94;
    if ( v43 > 0x40 )
    {
      v81 = v42;
      sub_C43690((__int64)&v86, 0, 0);
      v43 = v87;
      v42 = v81;
      if ( v81 == v87 )
        goto LABEL_93;
    }
    else
    {
      v86 = 0;
    }
    if ( v42 > 0x3F || v43 > 0x40 )
      sub_C43C90(&v86, v42, v43);
    else
      v86 |= 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v42 + 64 - (unsigned __int8)v43) << v42;
LABEL_93:
    v61 = *(_QWORD *)(v93.m128i_i64[0] + 3344);
    v103 = 257;
    v96 = (__m128i)v61;
    v97 = 0;
    v98 = 0;
    v99 = 0;
    v100 = 0;
    v101 = 0;
    v102 = 0;
    if ( !(unsigned __int8)sub_9AC230((__int64)v29, (__int64)&v86, &v96, 0) )
    {
      if ( v87 > 0x40 && v86 )
        j_j___libc_free_0_0(v86);
LABEL_36:
      if ( !sub_2B17B70(&v93, v29, v19) )
        break;
      goto LABEL_37;
    }
    if ( v87 > 0x40 && v86 )
      j_j___libc_free_0_0(v86);
LABEL_37:
    v19 = v95;
LABEL_38:
    v44 = (_BYTE *)v14[2];
    v30 = v14 + 2;
    if ( *v44 == 13 )
      goto LABEL_68;
    a5 = *(_BYTE *)(v93.m128i_i64[0] + 88) & 1;
    if ( (*(_BYTE *)(v93.m128i_i64[0] + 88) & 1) != 0 )
    {
      v45 = v93.m128i_i64[0] + 96;
      v46 = 3;
    }
    else
    {
      v56 = *(unsigned int *)(v93.m128i_i64[0] + 104);
      v45 = *(_QWORD *)(v93.m128i_i64[0] + 96);
      if ( !(_DWORD)v56 )
        goto LABEL_77;
      v46 = v56 - 1;
    }
    v9 = v46 & (((unsigned int)v44 >> 9) ^ ((unsigned int)v44 >> 4));
    a6 = v45 + 72 * v9;
    v47 = *(_QWORD *)a6;
    if ( v44 == *(_BYTE **)a6 )
      goto LABEL_42;
    v65 = 1;
    while ( v47 != -4096 )
    {
      v68 = v65 + 1;
      v9 = v46 & (unsigned int)(v65 + v9);
      a6 = v45 + 72LL * (unsigned int)v9;
      v47 = *(_QWORD *)a6;
      if ( v44 == *(_BYTE **)a6 )
        goto LABEL_42;
      v65 = v68;
    }
    if ( (_BYTE)a5 )
    {
      v59 = 288;
      goto LABEL_78;
    }
    v56 = *(unsigned int *)(v93.m128i_i64[0] + 104);
LABEL_77:
    v59 = 72 * v56;
LABEL_78:
    a6 = v45 + v59;
LABEL_42:
    v48 = 288;
    if ( !(_BYTE)a5 )
      v48 = 72LL * *(unsigned int *)(v93.m128i_i64[0] + 104);
    if ( a6 != v45 + v48 && *(_DWORD *)(a6 + 16) > 1u )
      break;
    v49 = *(_QWORD *)(v93.m128i_i64[0] + 3344);
    v50 = v14[2];
    v103 = 257;
    v96 = (__m128i)v49;
    v97 = 0;
    v98 = 0;
    v99 = 0;
    v100 = 0;
    v101 = 0;
    v102 = 0;
    v51 = sub_9AC470(v50, &v96, 0);
    if ( *(_BYTE *)v93.m128i_i64[1] && v51 )
      goto LABEL_49;
    v52 = *v19;
    v53 = *v94;
    if ( *v94 <= *v19 )
      goto LABEL_49;
    v87 = *v94;
    if ( v53 <= 0x40 )
    {
      v86 = 0;
LABEL_99:
      if ( v52 > 0x3F || v53 > 0x40 )
        sub_C43C90(&v86, v52, v53);
      else
        v86 |= 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v52 + 64 - (unsigned __int8)v53) << v52;
      goto LABEL_102;
    }
    v82 = v52;
    sub_C43690((__int64)&v86, 0, 0);
    v53 = v87;
    v52 = v82;
    if ( v82 != v87 )
      goto LABEL_99;
LABEL_102:
    v62 = *(_QWORD *)(v93.m128i_i64[0] + 3344);
    v97 = 0;
    v96 = (__m128i)v62;
    v98 = 0;
    v99 = 0;
    v100 = 0;
    v101 = 0;
    v102 = 0;
    v103 = 257;
    if ( (unsigned __int8)sub_9AC230((__int64)v44, (__int64)&v86, &v96, 0) )
    {
      if ( v87 > 0x40 && v86 )
        j_j___libc_free_0_0(v86);
      v19 = v95;
LABEL_68:
      if ( !sub_2B1EE10(v93.m128i_i64, (_BYTE *)v14[3], v19) )
        goto LABEL_69;
      goto LABEL_51;
    }
    if ( v87 > 0x40 && v86 )
      j_j___libc_free_0_0(v86);
LABEL_49:
    if ( !sub_2B17B70(&v93, v44, v19) )
      break;
    if ( !sub_2B1EE10(v93.m128i_i64, (_BYTE *)v14[3], v95) )
    {
LABEL_69:
      v14 += 3;
      goto LABEL_11;
    }
LABEL_51:
    v14 += 4;
    if ( v14 == v85 )
    {
      v16 = v84 - v14;
      goto LABEL_53;
    }
  }
  result = 0;
  if ( v84 != v30 )
    return result;
LABEL_27:
  v35 = *(_QWORD *)(a1 + 16);
  result = 1;
  if ( *(_DWORD *)(v35 + 104) != 3 )
    return result;
  v36 = *(_QWORD *)(v35 + 416);
  if ( v36 && *(_QWORD *)(v35 + 424) )
  {
    v37 = a1;
    v38 = sub_2B31EF0(*(_QWORD *)(a1 + 40), v36, *(char **)v35, *(unsigned int *)(v35 + 8), 0);
    if ( v38 )
    {
      v37 = a1;
      if ( (unsigned __int8)sub_2B69990(
                              *(_QWORD **)(a1 + 40),
                              v38,
                              **(_BYTE **)(a1 + 48),
                              *(unsigned int **)(a1 + 32),
                              *(_QWORD **)(a1 + 56),
                              *(_QWORD *)(a1 + 64),
                              *(_QWORD *)(a1 + 72),
                              *(_DWORD **)(a1 + 80),
                              *(_BYTE **)(a1 + 8),
                              **(_BYTE **)(a1 + 88)) )
      {
        sub_9C8C60(*(_QWORD *)(a1 + 56), *(_DWORD *)(*(_QWORD *)(a1 + 16) + 200LL));
        return 1;
      }
    }
    v35 = *(_QWORD *)(v37 + 16);
  }
  v96.m128i_i64[0] = 0;
  v96.m128i_i64[1] = (__int64)&v99;
  v97 = 4;
  LODWORD(v98) = 0;
  BYTE4(v98) = 1;
  v69 = *(char **)v35;
  v70 = *(_QWORD *)v35 + 8LL * *(unsigned int *)(v35 + 8);
  if ( *(_QWORD *)v35 == v70 )
    goto LABEL_155;
  do
  {
    while ( 1 )
    {
      v71 = *(__int64 ***)v69;
      if ( **(_BYTE **)v69 == 90 )
        break;
      v69 += 8;
      if ( (char *)v70 == v69 )
        goto LABEL_143;
    }
    v69 += 8;
    sub_2411830((__int64)&v93, (__int64)&v96, *(v71 - 8), v9, a5, a6);
  }
  while ( (char *)v70 != v69 );
LABEL_143:
  v35 = *(_QWORD *)(a1 + 16);
  if ( (unsigned int)(HIDWORD(v97) - v98) <= 2 )
  {
LABEL_155:
    v78 = *(_DWORD *)(v35 + 200);
    v79 = *(_QWORD *)(a1 + 56);
    v80 = *(unsigned int *)(v79 + 8);
    if ( v80 + 1 > (unsigned __int64)*(unsigned int *)(v79 + 12) )
    {
      sub_C8D5F0(*(_QWORD *)(a1 + 56), (const void *)(v79 + 16), v80 + 1, 4u, a5, a6);
      v80 = *(unsigned int *)(v79 + 8);
    }
    *(_DWORD *)(*(_QWORD *)v79 + 4 * v80) = v78;
    ++*(_DWORD *)(v79 + 8);
    goto LABEL_145;
  }
  v72 = *(_DWORD *)(v35 + 8);
  v73 = *(_QWORD ***)(**(_QWORD **)v35 + 8LL);
  v74 = sub_2B08680((__int64)v73, v72);
  v75 = sub_2B1F810(*(_QWORD *)a1, v74, 0xFFFFFFFF);
  v76 = sub_BCCE00(*v73, **(_DWORD **)(a1 + 32));
  v77 = sub_2B08680(v76, v72);
  if ( v75 >= (unsigned int)sub_2B1F810(*(_QWORD *)a1, v77, 0xFFFFFFFF) )
  {
    v35 = *(_QWORD *)(a1 + 16);
    goto LABEL_155;
  }
LABEL_145:
  if ( !BYTE4(v98) )
    _libc_free(v96.m128i_u64[1]);
  return 1;
}
