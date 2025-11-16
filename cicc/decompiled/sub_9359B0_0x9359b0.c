// Function: sub_9359B0
// Address: 0x9359b0
//
__int64 __fastcall sub_9359B0(__int64 a1, _QWORD *a2)
{
  unsigned int v2; // r13d
  __int64 v3; // r12
  __int64 v5; // rsi
  __int64 v6; // rcx
  __int64 i; // r14
  _BYTE *v8; // rsi
  __int64 v9; // rax
  _BYTE *v10; // rsi
  _BYTE *v11; // rax
  __int64 v12; // rax
  _BYTE *v13; // rsi
  unsigned int v14; // esi
  unsigned int v15; // r9d
  __int64 v16; // r8
  unsigned int v17; // edx
  unsigned int v18; // edi
  unsigned __int64 *v19; // rax
  unsigned __int64 v20; // rcx
  _QWORD *v21; // r10
  void **v22; // r14
  _BYTE *v23; // rdi
  unsigned __int64 v24; // rax
  char *v25; // rdx
  char *v26; // rsi
  size_t v27; // r15
  _BYTE *v28; // r8
  char *v29; // r9
  char *v30; // rcx
  _QWORD *v31; // r14
  __int64 v32; // rax
  __int64 v33; // rbx
  __int64 v34; // rax
  __int64 v35; // r15
  unsigned int *v36; // r15
  unsigned int *v37; // r12
  __int64 v38; // rdx
  __int64 v39; // rsi
  __int64 v40; // r15
  __int64 v41; // r13
  __int64 v42; // rdx
  __int64 v43; // rsi
  _QWORD *v44; // rax
  __int64 result; // rax
  char *v46; // rsi
  __int64 v47; // rax
  char *v48; // rcx
  int v49; // r10d
  unsigned __int64 *v50; // r14
  int v51; // eax
  int v52; // eax
  _QWORD *v53; // r10
  void **v54; // r11
  void **v55; // r14
  int v56; // r15d
  int v57; // eax
  int v58; // eax
  int v59; // ecx
  __int64 v60; // rsi
  unsigned int v61; // edx
  unsigned __int64 v62; // rdi
  int v63; // r9d
  unsigned __int64 *v64; // r8
  int v65; // eax
  int v66; // eax
  __int64 v67; // rsi
  unsigned int v68; // edx
  void *v69; // rdi
  int v70; // r9d
  void **v71; // r8
  int v72; // eax
  int v73; // eax
  __int64 v74; // rdi
  int v75; // r9d
  unsigned int v76; // edx
  _QWORD *v77; // rsi
  int v78; // eax
  int v79; // ecx
  int v80; // r9d
  __int64 v81; // rdi
  unsigned int v82; // edx
  _QWORD *v83; // rsi
  int v84; // r14d
  void **v85; // r15
  char *v86; // [rsp+0h] [rbp-D0h]
  void **v87; // [rsp+0h] [rbp-D0h]
  __int64 v88; // [rsp+8h] [rbp-C8h]
  char *v89; // [rsp+8h] [rbp-C8h]
  char *v90; // [rsp+8h] [rbp-C8h]
  unsigned int v91; // [rsp+8h] [rbp-C8h]
  unsigned int v92; // [rsp+8h] [rbp-C8h]
  unsigned int v93; // [rsp+8h] [rbp-C8h]
  __m128i *v94; // [rsp+10h] [rbp-C0h]
  __int64 v95; // [rsp+10h] [rbp-C0h]
  __int64 v97; // [rsp+20h] [rbp-B0h] BYREF
  _BYTE *v98; // [rsp+28h] [rbp-A8h] BYREF
  __int64 v99; // [rsp+30h] [rbp-A0h] BYREF
  _BYTE *v100; // [rsp+38h] [rbp-98h]
  _BYTE *v101; // [rsp+40h] [rbp-90h]
  void *src; // [rsp+50h] [rbp-80h] BYREF
  _BYTE *v103; // [rsp+58h] [rbp-78h]
  _BYTE *v104; // [rsp+60h] [rbp-70h]
  char v105[32]; // [rsp+70h] [rbp-60h] BYREF
  __int16 v106; // [rsp+90h] [rbp-40h]

  v2 = 0;
  v3 = a1;
  v5 = a2[6];
  v99 = 0;
  v100 = 0;
  v101 = 0;
  src = 0;
  v103 = 0;
  v104 = 0;
  v94 = sub_92F410(a1, v5);
  for ( i = *(_QWORD *)(a2[10] + 16LL); i; v103 = v10 + 8 )
  {
    while ( 1 )
    {
      v11 = (_BYTE *)sub_91FFA0(*(_QWORD *)(a1 + 32), *(const __m128i **)(i + 8), 0, v6);
      v8 = v100;
      if ( *v11 != 17 )
        v11 = 0;
      v98 = v11;
      if ( v100 == v101 )
      {
        sub_931810((__int64)&v99, v100, &v98);
      }
      else
      {
        if ( v100 )
        {
          *(_QWORD *)v100 = v11;
          v8 = v100;
        }
        v100 = v8 + 8;
      }
      v9 = sub_945CA0(a1, "switch_case.target", 0, 0);
      v10 = v103;
      v97 = v9;
      if ( v103 != v104 )
        break;
      ++v2;
      sub_9319A0((__int64)&src, v103, &v97);
      i = *(_QWORD *)(i + 32);
      if ( !i )
        goto LABEL_15;
    }
    if ( v103 )
    {
      *(_QWORD *)v103 = v9;
      v10 = v103;
    }
    i = *(_QWORD *)(i + 32);
    ++v2;
  }
LABEL_15:
  v12 = sub_945CA0(a1, "switch_case.default_target", 0, 0);
  v13 = v103;
  v97 = v12;
  if ( v103 == v104 )
  {
    sub_9319A0((__int64)&src, v103, &v97);
  }
  else
  {
    if ( v103 )
    {
      *(_QWORD *)v103 = v12;
      v13 = v103;
    }
    v103 = v13 + 8;
  }
  v14 = *(_DWORD *)(a1 + 520);
  if ( !v14 )
  {
    ++*(_QWORD *)(a1 + 496);
    goto LABEL_90;
  }
  v15 = v14 - 1;
  v16 = *(_QWORD *)(a1 + 504);
  v17 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
  v18 = (v14 - 1) & v17;
  v19 = (unsigned __int64 *)(v16 + 32LL * v18);
  v20 = *v19;
  if ( (_QWORD *)*v19 == a2 )
  {
    v21 = v19 + 1;
    v22 = (void **)(v19 + 1);
    if ( v19 + 1 == (unsigned __int64 *)&src )
      goto LABEL_31;
LABEL_22:
    v23 = (_BYTE *)v19[1];
    v24 = v19[3] - (_QWORD)v23;
    goto LABEL_23;
  }
  v91 = (v14 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v53 = (_QWORD *)*v19;
  v54 = (void **)(v16 + 32LL * (v15 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4))));
  v55 = 0;
  v56 = 1;
  while ( 1 )
  {
    if ( v53 == (_QWORD *)-4096LL )
    {
      v57 = *(_DWORD *)(v3 + 512);
      if ( !v55 )
        v55 = v54;
      ++*(_QWORD *)(v3 + 496);
      v20 = (unsigned int)(v57 + 1);
      if ( 4 * (int)v20 < 3 * v14 )
      {
        if ( v14 - *(_DWORD *)(v3 + 516) - (unsigned int)v20 > v14 >> 3 )
          goto LABEL_77;
        v92 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
        sub_935440(v3 + 496, v14);
        v72 = *(_DWORD *)(v3 + 520);
        if ( !v72 )
          goto LABEL_137;
        v73 = v72 - 1;
        v74 = *(_QWORD *)(v3 + 504);
        v75 = 1;
        v71 = 0;
        v76 = v73 & v92;
        v20 = (unsigned int)(*(_DWORD *)(v3 + 512) + 1);
        v55 = (void **)(v74 + 32LL * (v73 & v92));
        v77 = *v55;
        if ( *v55 == a2 )
          goto LABEL_77;
        while ( v77 != (_QWORD *)-4096LL )
        {
          if ( !v71 && v77 == (_QWORD *)-8192LL )
            v71 = v55;
          v76 = v73 & (v75 + v76);
          v55 = (void **)(v74 + 32LL * v76);
          v77 = *v55;
          if ( *v55 == a2 )
            goto LABEL_77;
          ++v75;
        }
        goto LABEL_94;
      }
LABEL_90:
      sub_935440(v3 + 496, 2 * v14);
      v65 = *(_DWORD *)(v3 + 520);
      if ( !v65 )
        goto LABEL_137;
      v66 = v65 - 1;
      v67 = *(_QWORD *)(v3 + 504);
      v68 = v66 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v20 = (unsigned int)(*(_DWORD *)(v3 + 512) + 1);
      v55 = (void **)(v67 + 32LL * v68);
      v69 = *v55;
      if ( *v55 == a2 )
        goto LABEL_77;
      v70 = 1;
      v71 = 0;
      while ( v69 != (void *)-4096LL )
      {
        if ( v69 == (void *)-8192LL && !v71 )
          v71 = v55;
        v68 = v66 & (v70 + v68);
        v55 = (void **)(v67 + 32LL * v68);
        v69 = *v55;
        if ( *v55 == a2 )
          goto LABEL_77;
        ++v70;
      }
LABEL_94:
      if ( v71 )
        v55 = v71;
LABEL_77:
      *(_DWORD *)(v3 + 512) = v20;
      if ( *v55 != (void *)-4096LL )
        --*(_DWORD *)(v3 + 516);
      v55[1] = 0;
      v22 = v55 + 1;
      v22[1] = 0;
      *(v22 - 1) = a2;
      v22[2] = 0;
      if ( v22 == &src )
        goto LABEL_29;
      v24 = 0;
      v23 = 0;
LABEL_23:
      v25 = v103;
      v26 = (char *)src;
      v27 = v103 - (_BYTE *)src;
      if ( v103 - (_BYTE *)src > v24 )
      {
        if ( v27 )
        {
          if ( v27 > 0x7FFFFFFFFFFFFFF8LL )
            sub_4261EA(v23, src, v103, v20);
          v86 = (char *)src;
          v89 = v103;
          v47 = sub_22077B0(v103 - (_BYTE *)src);
          v25 = v89;
          v26 = v86;
          v48 = (char *)v47;
        }
        else
        {
          v48 = 0;
        }
        if ( v25 != v26 )
          v48 = (char *)memcpy(v48, v26, v27);
        if ( *v22 )
        {
          v90 = v48;
          j_j___libc_free_0(*v22, (_BYTE *)v22[2] - (_BYTE *)*v22);
          v48 = v90;
        }
        *v22 = v48;
        v30 = &v48[v27];
        v22[2] = v30;
        goto LABEL_28;
      }
      v28 = v22[1];
      v29 = (char *)(v28 - v23);
      if ( v27 > v28 - v23 )
      {
        if ( v29 )
        {
          memmove(v23, src, (_BYTE *)v22[1] - v23);
          v28 = v22[1];
          v23 = *v22;
          v25 = v103;
          v26 = (char *)src;
          v29 = (char *)(v28 - (_BYTE *)*v22);
        }
        v46 = &v26[(_QWORD)v29];
        if ( v46 != v25 )
        {
          memmove(v28, v46, v25 - v46);
          v30 = (char *)*v22 + v27;
          goto LABEL_28;
        }
      }
      else if ( v103 != src )
      {
        memmove(v23, src, v103 - (_BYTE *)src);
        v23 = *v22;
      }
      v30 = &v23[v27];
LABEL_28:
      v22[1] = v30;
LABEL_29:
      v14 = *(_DWORD *)(v3 + 520);
      if ( v14 )
      {
        v15 = v14 - 1;
        v16 = *(_QWORD *)(v3 + 504);
        v17 = ((unsigned int)a2 >> 4) ^ ((unsigned int)a2 >> 9);
        v18 = (v14 - 1) & v17;
        v19 = (unsigned __int64 *)(v16 + 32LL * v18);
        v20 = *v19;
        v21 = v19 + 1;
        if ( (_QWORD *)*v19 == a2 )
          goto LABEL_31;
        goto LABEL_61;
      }
      ++*(_QWORD *)(v3 + 496);
      goto LABEL_82;
    }
    if ( v55 || v53 != (_QWORD *)-8192LL )
      v54 = v55;
    v84 = v56 + 1;
    v91 = v15 & (v91 + v56);
    v85 = (void **)(v16 + 32LL * v91);
    v87 = v85;
    v53 = *v85;
    if ( *v85 == a2 )
      break;
    v56 = v84;
    v55 = v54;
    v54 = v87;
  }
  v22 = v85 + 1;
  if ( v85 + 1 != &src )
  {
    v19 = (unsigned __int64 *)(v16 + 32LL * v91);
    goto LABEL_22;
  }
LABEL_61:
  v49 = 1;
  v50 = 0;
  while ( 2 )
  {
    if ( v20 == -4096 )
    {
      if ( !v50 )
        v50 = v19;
      v51 = *(_DWORD *)(v3 + 512);
      ++*(_QWORD *)(v3 + 496);
      v52 = v51 + 1;
      if ( 4 * v52 < 3 * v14 )
      {
        if ( v14 - (v52 + *(_DWORD *)(v3 + 516)) > v14 >> 3 )
        {
LABEL_67:
          *(_DWORD *)(v3 + 512) = v52;
          if ( *v50 != -4096 )
            --*(_DWORD *)(v3 + 516);
          v50[1] = 0;
          v31 = v50 + 1;
          v31[1] = 0;
          *(v31 - 1) = a2;
          v31[2] = 0;
          goto LABEL_32;
        }
        v93 = v17;
        sub_935440(v3 + 496, v14);
        v78 = *(_DWORD *)(v3 + 520);
        if ( v78 )
        {
          v79 = v78 - 1;
          v80 = 1;
          v64 = 0;
          v81 = *(_QWORD *)(v3 + 504);
          v82 = v79 & v93;
          v52 = *(_DWORD *)(v3 + 512) + 1;
          v50 = (unsigned __int64 *)(v81 + 32LL * (v79 & v93));
          v83 = (_QWORD *)*v50;
          if ( (_QWORD *)*v50 == a2 )
            goto LABEL_67;
          while ( v83 != (_QWORD *)-4096LL )
          {
            if ( !v64 && v83 == (_QWORD *)-8192LL )
              v64 = v50;
            v82 = v79 & (v80 + v82);
            v50 = (unsigned __int64 *)(v81 + 32LL * v82);
            v83 = (_QWORD *)*v50;
            if ( (_QWORD *)*v50 == a2 )
              goto LABEL_67;
            ++v80;
          }
          goto LABEL_86;
        }
LABEL_137:
        ++*(_DWORD *)(v3 + 512);
        BUG();
      }
LABEL_82:
      sub_935440(v3 + 496, 2 * v14);
      v58 = *(_DWORD *)(v3 + 520);
      if ( v58 )
      {
        v59 = v58 - 1;
        v60 = *(_QWORD *)(v3 + 504);
        v61 = (v58 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v52 = *(_DWORD *)(v3 + 512) + 1;
        v50 = (unsigned __int64 *)(v60 + 32LL * v61);
        v62 = *v50;
        if ( (_QWORD *)*v50 == a2 )
          goto LABEL_67;
        v63 = 1;
        v64 = 0;
        while ( v62 != -4096 )
        {
          if ( v62 == -8192 && !v64 )
            v64 = v50;
          v61 = v59 & (v63 + v61);
          v50 = (unsigned __int64 *)(v60 + 32LL * v61);
          v62 = *v50;
          if ( (_QWORD *)*v50 == a2 )
            goto LABEL_67;
          ++v63;
        }
LABEL_86:
        if ( v64 )
          v50 = v64;
        goto LABEL_67;
      }
      goto LABEL_137;
    }
    if ( !v50 && v20 == -8192 )
      v50 = v19;
    v18 = v15 & (v49 + v18);
    v19 = (unsigned __int64 *)(v16 + 32LL * v18);
    v20 = *v19;
    if ( (_QWORD *)*v19 != a2 )
    {
      ++v49;
      continue;
    }
    break;
  }
  v21 = v19 + 1;
LABEL_31:
  v31 = v21;
LABEL_32:
  v106 = 257;
  v88 = v97;
  v32 = sub_BD2DA0(80);
  v33 = v32;
  if ( v32 )
    sub_B53A60(v32, v94, v88, v2, 0, 0);
  (*(void (__fastcall **)(_QWORD, __int64, char *, _QWORD, _QWORD))(**(_QWORD **)(v3 + 136) + 16LL))(
    *(_QWORD *)(v3 + 136),
    v33,
    v105,
    *(_QWORD *)(v3 + 104),
    *(_QWORD *)(v3 + 112));
  v34 = *(_QWORD *)(v3 + 48);
  v35 = 16LL * *(unsigned int *)(v3 + 56);
  if ( v34 != v34 + v35 )
  {
    v95 = v3;
    v36 = (unsigned int *)(v34 + v35);
    v37 = *(unsigned int **)(v3 + 48);
    do
    {
      v38 = *((_QWORD *)v37 + 1);
      v39 = *v37;
      v37 += 4;
      sub_B99FD0(v33, v39, v38);
    }
    while ( v36 != v37 );
    v3 = v95;
  }
  if ( v2 )
  {
    v40 = 0;
    v41 = 8LL * (v2 - 1) + 8;
    do
    {
      v42 = *(_QWORD *)(*v31 + v40);
      v43 = *(_QWORD *)(v99 + v40);
      v40 += 8;
      sub_B53E30(v33, v43, v42);
    }
    while ( v41 != v40 );
  }
  v44 = (_QWORD *)sub_945CA0(v3, "switch_child_entry", 0, 0);
  sub_92FEA0(v3, v44, 0);
  sub_9363D0(v3, a2[9]);
  result = a2[10];
  if ( !*(_QWORD *)(result + 8) )
    result = sub_92FEA0(v3, *((_QWORD **)v103 - 1), 0);
  if ( src )
    result = j_j___libc_free_0(src, v104 - (_BYTE *)src);
  if ( v99 )
    return j_j___libc_free_0(v99, &v101[-v99]);
  return result;
}
