// Function: sub_26E9DC0
// Address: 0x26e9dc0
//
void __fastcall sub_26E9DC0(_QWORD *a1, __int64 a2, __int64 a3)
{
  unsigned int v4; // edx
  __int64 *v5; // rsi
  __int64 v6; // r8
  __int64 v7; // rcx
  __int64 v8; // rax
  unsigned int v9; // edx
  __int64 *v10; // rsi
  __int64 v11; // r8
  unsigned __int64 v12; // r12
  __int64 v13; // rcx
  __int64 v14; // rax
  unsigned __int64 v15; // rdi
  int v16; // r14d
  __int64 v17; // rax
  _QWORD *v18; // r8
  __int64 v19; // rbx
  _QWORD *v20; // rax
  _QWORD *v21; // rsi
  int *v22; // r13
  int v23; // esi
  __int64 v24; // rbx
  char v25; // al
  __int64 v26; // rax
  unsigned int v27; // r9d
  unsigned __int64 v28; // rdi
  __int64 v29; // r8
  unsigned int v30; // r13d
  unsigned __int64 v31; // rdx
  _QWORD *v32; // r10
  __int64 v33; // r14
  _QWORD *v34; // rax
  _QWORD *v35; // rsi
  unsigned int *v36; // r12
  __int64 v37; // rax
  _QWORD *v38; // r12
  __int64 v39; // rdx
  __int64 v40; // rsi
  _QWORD *v41; // rdi
  char v42; // al
  unsigned __int64 v43; // rdx
  unsigned __int64 v44; // r8
  unsigned __int64 v45; // r10
  _QWORD *v46; // r11
  _QWORD *v47; // rax
  _QWORD *v48; // rdx
  size_t v49; // r14
  void *v50; // rax
  _QWORD *v51; // rax
  _QWORD *v52; // r9
  _QWORD *v53; // rsi
  unsigned __int64 v54; // rdi
  _QWORD *v55; // rcx
  unsigned __int64 v56; // rdx
  _QWORD **v57; // rax
  unsigned __int64 v58; // rdi
  __int64 v59; // rdx
  char *v60; // rax
  __int64 v61; // rdx
  __m128i *v62; // rax
  __m128i *v63; // rax
  unsigned __int64 v64; // rcx
  __int64 v65; // r14
  size_t v66; // rax
  __int64 v67; // rax
  _QWORD *v68; // r13
  __int64 v69; // rdx
  char v70; // al
  unsigned __int64 v71; // r8
  _QWORD *v72; // r9
  _QWORD **v73; // rax
  _QWORD *v74; // rdx
  int v75; // esi
  int v76; // r9d
  __int64 v77; // rbx
  void *v78; // rax
  size_t v79; // rdx
  _QWORD *v80; // rbx
  _QWORD *v81; // rax
  _QWORD *v82; // rsi
  unsigned __int64 v83; // rdi
  _QWORD *v84; // rcx
  unsigned __int64 v85; // rdx
  _QWORD **v86; // rax
  unsigned __int64 v87; // rdi
  __int64 v88; // rdx
  int v89; // r9d
  __int64 v90; // [rsp+0h] [rbp-110h]
  __int64 v91; // [rsp+8h] [rbp-108h]
  unsigned __int64 v92; // [rsp+10h] [rbp-100h]
  _QWORD *v93; // [rsp+18h] [rbp-F8h]
  __int64 v94; // [rsp+20h] [rbp-F0h]
  _QWORD *v97; // [rsp+38h] [rbp-D8h]
  unsigned __int64 v98; // [rsp+48h] [rbp-C8h]
  unsigned __int64 v99; // [rsp+48h] [rbp-C8h]
  unsigned __int64 v100; // [rsp+48h] [rbp-C8h]
  __int64 v101; // [rsp+50h] [rbp-C0h]
  __int64 v102; // [rsp+58h] [rbp-B8h]
  unsigned __int64 v103; // [rsp+58h] [rbp-B8h]
  _QWORD *v104; // [rsp+58h] [rbp-B8h]
  unsigned __int64 v105[2]; // [rsp+60h] [rbp-B0h] BYREF
  __m128i v106; // [rsp+70h] [rbp-A0h] BYREF
  __int64 v107[2]; // [rsp+80h] [rbp-90h] BYREF
  _QWORD v108[2]; // [rsp+90h] [rbp-80h] BYREF
  __int16 v109; // [rsp+A0h] [rbp-70h]
  __m128i *v110; // [rsp+B0h] [rbp-60h] BYREF
  __int64 v111; // [rsp+B8h] [rbp-58h]
  __m128i v112; // [rsp+C0h] [rbp-50h] BYREF
  int v113; // [rsp+D0h] [rbp-40h]
  __int64 *v114; // [rsp+D8h] [rbp-38h]

  v97 = a1 + 15;
  v91 = sub_B2BE50(*a1);
  v90 = *(_QWORD *)(*a1 + 40LL);
  v94 = *a1 + 72LL;
  v101 = *(_QWORD *)(*a1 + 80LL);
  if ( v101 == v94 )
    return;
  while ( 1 )
  {
    v12 = v101 - 24;
    if ( !v101 )
      v12 = 0;
    v13 = *(_QWORD *)(a2 + 8);
    v14 = *(unsigned int *)(a2 + 24);
    if ( (_DWORD)v14 )
    {
      v4 = (v14 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
      v5 = (__int64 *)(v13 + 8LL * v4);
      v6 = *v5;
      if ( v12 == *v5 )
      {
LABEL_4:
        if ( v5 != (__int64 *)(v13 + 8 * v14) )
          goto LABEL_5;
      }
      else
      {
        v75 = 1;
        while ( v6 != -4096 )
        {
          v76 = v75 + 1;
          v4 = (v14 - 1) & (v75 + v4);
          v5 = (__int64 *)(v13 + 8LL * v4);
          v6 = *v5;
          if ( v12 == *v5 )
            goto LABEL_4;
          v75 = v76;
        }
      }
    }
    v15 = a1[7];
    v16 = *((_DWORD *)a1 + 40) + 1;
    v17 = a1[6];
    *((_DWORD *)a1 + 40) = v16;
    v18 = *(_QWORD **)(v17 + 8 * (v12 % v15));
    v19 = v12 % v15;
    if ( !v18 )
      goto LABEL_76;
    v20 = (_QWORD *)*v18;
    if ( v12 != *(_QWORD *)(*v18 + 8LL) )
    {
      while ( 1 )
      {
        v21 = (_QWORD *)*v20;
        if ( !*v20 )
          break;
        v18 = v20;
        if ( v12 % v15 != v21[1] % v15 )
          break;
        v20 = (_QWORD *)*v20;
        if ( v12 == v21[1] )
          goto LABEL_17;
      }
LABEL_76:
      v67 = sub_22077B0(0x18u);
      v68 = (_QWORD *)v67;
      if ( v67 )
        *(_QWORD *)v67 = 0;
      v69 = a1[9];
      v40 = a1[7];
      *(_QWORD *)(v67 + 8) = v12;
      v41 = a1 + 10;
      *(_DWORD *)(v67 + 16) = 0;
      v70 = sub_222DA10((__int64)(a1 + 10), v40, v69, 1);
      v71 = v43;
      if ( !v70 )
      {
        v72 = (_QWORD *)a1[6];
        v73 = (_QWORD **)&v72[v19];
        v74 = (_QWORD *)v72[v19];
        if ( v74 )
        {
LABEL_80:
          *v68 = *v74;
          **v73 = v68;
LABEL_81:
          ++a1[9];
          v22 = (int *)(v68 + 2);
          goto LABEL_18;
        }
LABEL_102:
        v88 = a1[8];
        a1[8] = v68;
        *v68 = v88;
        if ( v88 )
        {
          v72[*(_QWORD *)(v88 + 8) % a1[7]] = v68;
          v73 = (_QWORD **)(v19 * 8 + a1[6]);
        }
        *v73 = a1 + 8;
        goto LABEL_81;
      }
      if ( v43 == 1 )
      {
        v72 = a1 + 12;
        a1[12] = 0;
        v80 = a1 + 12;
      }
      else
      {
        if ( v43 > 0xFFFFFFFFFFFFFFFLL )
LABEL_112:
          sub_4261EA(v41, v40, v43);
        v77 = 8 * v43;
        v103 = v43;
        v78 = (void *)sub_22077B0(8 * v43);
        v79 = v77;
        v80 = a1 + 12;
        v81 = memset(v78, 0, v79);
        v71 = v103;
        v72 = v81;
      }
      v82 = (_QWORD *)a1[8];
      a1[8] = 0;
      if ( !v82 )
      {
LABEL_99:
        v87 = a1[6];
        if ( (_QWORD *)v87 != v80 )
        {
          v100 = v71;
          v104 = v72;
          j_j___libc_free_0(v87);
          v71 = v100;
          v72 = v104;
        }
        a1[7] = v71;
        a1[6] = v72;
        v19 = v12 % v71;
        v73 = (_QWORD **)&v72[v19];
        v74 = (_QWORD *)v72[v19];
        if ( v74 )
          goto LABEL_80;
        goto LABEL_102;
      }
      v83 = 0;
      while ( 1 )
      {
        while ( 1 )
        {
          v84 = v82;
          v82 = (_QWORD *)*v82;
          v85 = v84[1] % v71;
          v86 = (_QWORD **)&v72[v85];
          if ( !*v86 )
            break;
          *v84 = **v86;
          **v86 = v84;
LABEL_95:
          if ( !v82 )
            goto LABEL_99;
        }
        *v84 = a1[8];
        a1[8] = v84;
        *v86 = a1 + 8;
        if ( !*v84 )
        {
          v83 = v85;
          goto LABEL_95;
        }
        v72[v83] = v84;
        v83 = v85;
        if ( !v82 )
          goto LABEL_99;
      }
    }
LABEL_17:
    v22 = (int *)(*v18 + 16LL);
    if ( !*v18 )
      goto LABEL_76;
LABEL_18:
    *v22 = v16;
LABEL_5:
    v7 = *(_QWORD *)(a3 + 8);
    v8 = *(unsigned int *)(a3 + 24);
    if ( !(_DWORD)v8 )
      goto LABEL_21;
    v9 = (v8 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
    v10 = (__int64 *)(v7 + 8LL * v9);
    v11 = *v10;
    if ( v12 != *v10 )
      break;
LABEL_7:
    if ( v10 == (__int64 *)(v7 + 8 * v8) )
      goto LABEL_21;
LABEL_8:
    v101 = *(_QWORD *)(v101 + 8);
    if ( v94 == v101 )
      return;
  }
  v23 = 1;
  while ( v11 != -4096 )
  {
    v89 = v23 + 1;
    v9 = (v8 - 1) & (v23 + v9);
    v10 = (__int64 *)(v7 + 8LL * v9);
    v11 = *v10;
    if ( v12 == *v10 )
      goto LABEL_7;
    v23 = v89;
  }
LABEL_21:
  v24 = *(_QWORD *)(v12 + 56);
  v102 = v12 + 48;
  if ( v12 + 48 == v24 )
    goto LABEL_8;
  while ( 2 )
  {
    if ( !v24 )
      BUG();
    v25 = *(_BYTE *)(v24 - 24);
    if ( v25 == 85 )
    {
      v26 = *(_QWORD *)(v24 - 56);
      if ( !v26 || *(_BYTE *)v26 || *(_QWORD *)(v26 + 24) != *(_QWORD *)(v24 + 56) || (*(_BYTE *)(v26 + 33) & 0x20) == 0 )
        goto LABEL_31;
LABEL_25:
      v24 = *(_QWORD *)(v24 + 8);
      if ( v102 == v24 )
        goto LABEL_8;
      continue;
    }
    break;
  }
  if ( v25 != 34 && v25 != 40 )
    goto LABEL_25;
LABEL_31:
  v27 = *((_DWORD *)a1 + 40);
  if ( v27 <= 0xFFFE )
  {
    v28 = a1[14];
    v29 = v24 - 24;
    v30 = v27 + 1;
    *((_DWORD *)a1 + 40) = v27 + 1;
    v31 = (v24 - 24) % v28;
    v32 = *(_QWORD **)(a1[13] + 8 * v31);
    v33 = v31;
    if ( !v32 )
      goto LABEL_39;
    v34 = (_QWORD *)*v32;
    if ( v29 == *(_QWORD *)(*v32 + 8LL) )
    {
LABEL_37:
      v36 = (unsigned int *)(*v32 + 16LL);
      if ( !*v32 )
        goto LABEL_39;
    }
    else
    {
      while ( 1 )
      {
        v35 = (_QWORD *)*v34;
        if ( !*v34 )
          break;
        v32 = v34;
        if ( (v24 - 24) % v28 != v35[1] % v28 )
          break;
        v34 = (_QWORD *)*v34;
        if ( v29 == v35[1] )
          goto LABEL_37;
      }
LABEL_39:
      v37 = sub_22077B0(0x18u);
      v38 = (_QWORD *)v37;
      if ( v37 )
        *(_QWORD *)v37 = 0;
      *(_QWORD *)(v37 + 8) = v24 - 24;
      v39 = a1[16];
      *(_DWORD *)(v37 + 16) = 0;
      v40 = a1[14];
      v41 = a1 + 17;
      v42 = sub_222DA10((__int64)(a1 + 17), v40, v39, 1);
      v44 = v24 - 24;
      v45 = v43;
      if ( !v42 )
      {
        v46 = (_QWORD *)a1[13];
        v47 = &v46[v33];
        v48 = (_QWORD *)v46[v33];
        if ( v48 )
          goto LABEL_43;
LABEL_58:
        v59 = a1[15];
        a1[15] = v38;
        *v38 = v59;
        if ( v59 )
        {
          v46[*(_QWORD *)(v59 + 8) % a1[14]] = v38;
          v47 = (_QWORD *)(v33 * 8 + a1[13]);
        }
        *v47 = v97;
        goto LABEL_44;
      }
      if ( v43 == 1 )
      {
        v46 = a1 + 19;
        a1[19] = 0;
        v52 = a1 + 19;
      }
      else
      {
        if ( v43 > 0xFFFFFFFFFFFFFFFLL )
          goto LABEL_112;
        v49 = 8 * v43;
        v98 = v43;
        v50 = (void *)sub_22077B0(8 * v43);
        v51 = memset(v50, 0, v49);
        v44 = v24 - 24;
        v45 = v98;
        v52 = a1 + 19;
        v46 = v51;
      }
      v53 = (_QWORD *)a1[15];
      a1[15] = 0;
      if ( v53 )
      {
        v54 = 0;
        do
        {
          while ( 1 )
          {
            v55 = v53;
            v53 = (_QWORD *)*v53;
            v56 = v55[1] % v45;
            v57 = (_QWORD **)&v46[v56];
            if ( !*v57 )
              break;
            *v55 = **v57;
            **v57 = v55;
LABEL_51:
            if ( !v53 )
              goto LABEL_55;
          }
          *v55 = a1[15];
          a1[15] = v55;
          *v57 = v97;
          if ( !*v55 )
          {
            v54 = v56;
            goto LABEL_51;
          }
          v46[v54] = v55;
          v54 = v56;
        }
        while ( v53 );
      }
LABEL_55:
      v58 = a1[13];
      if ( v52 != (_QWORD *)v58 )
      {
        v92 = v45;
        v93 = v46;
        v99 = v44;
        j_j___libc_free_0(v58);
        v45 = v92;
        v46 = v93;
        v44 = v99;
      }
      a1[14] = v45;
      a1[13] = v46;
      v33 = v44 % v45;
      v47 = &v46[v33];
      v48 = (_QWORD *)v46[v33];
      if ( !v48 )
        goto LABEL_58;
LABEL_43:
      *v38 = *v48;
      *(_QWORD *)*v47 = v38;
LABEL_44:
      ++a1[16];
      v36 = (unsigned int *)(v38 + 2);
    }
    *v36 = v30;
    goto LABEL_25;
  }
  v60 = (char *)sub_BD5D20(*a1);
  v107[0] = (__int64)v108;
  sub_26E9140(v107, v60, (__int64)&v60[v61]);
  v62 = (__m128i *)sub_2241130((unsigned __int64 *)v107, 0, 0, "Pseudo instrumentation incomplete for ", 0x26u);
  v110 = &v112;
  if ( (__m128i *)v62->m128i_i64[0] == &v62[1] )
  {
    v112 = _mm_loadu_si128(v62 + 1);
  }
  else
  {
    v110 = (__m128i *)v62->m128i_i64[0];
    v112.m128i_i64[0] = v62[1].m128i_i64[0];
  }
  v111 = v62->m128i_i64[1];
  v62->m128i_i64[0] = (__int64)v62[1].m128i_i64;
  v62->m128i_i64[1] = 0;
  v62[1].m128i_i8[0] = 0;
  if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v111) <= 0x16 )
    sub_4262D8((__int64)"basic_string::append");
  v63 = (__m128i *)sub_2241490((unsigned __int64 *)&v110, " because it's too large", 0x17u);
  v105[0] = (unsigned __int64)&v106;
  if ( (__m128i *)v63->m128i_i64[0] == &v63[1] )
  {
    v106 = _mm_loadu_si128(v63 + 1);
  }
  else
  {
    v105[0] = v63->m128i_i64[0];
    v106.m128i_i64[0] = v63[1].m128i_i64[0];
  }
  v64 = v63->m128i_u64[1];
  v63[1].m128i_i8[0] = 0;
  v105[1] = v64;
  v63->m128i_i64[0] = (__int64)v63[1].m128i_i64;
  v63->m128i_i64[1] = 0;
  if ( v110 != &v112 )
    j_j___libc_free_0((unsigned __int64)v110);
  if ( (_QWORD *)v107[0] != v108 )
    j_j___libc_free_0(v107[0]);
  v109 = 260;
  v107[0] = (__int64)v105;
  v65 = *(_QWORD *)(v90 + 168);
  v66 = 0;
  if ( v65 )
    v66 = strlen(*(const char **)(v90 + 168));
  v112.m128i_i64[0] = v65;
  v111 = 0x10000000CLL;
  v112.m128i_i64[1] = v66;
  v110 = (__m128i *)&unk_49D9C78;
  v113 = 0;
  v114 = v107;
  sub_B6EB20(v91, (__int64)&v110);
  if ( (__m128i *)v105[0] != &v106 )
    j_j___libc_free_0(v105[0]);
}
