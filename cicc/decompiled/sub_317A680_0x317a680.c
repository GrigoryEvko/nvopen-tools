// Function: sub_317A680
// Address: 0x317a680
//
unsigned __int64 __fastcall sub_317A680(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v10; // rcx
  __int64 v11; // rsi
  unsigned int v12; // eax
  __int64 v13; // rdx
  int v15; // r10d
  __int64 v16; // rdx
  unsigned __int8 *v17; // rsi
  __int64 *v18; // rdi
  __m128i v19; // xmm1
  unsigned __int8 v20; // al
  __int64 v21; // r9
  int v22; // edx
  unsigned int v23; // esi
  __int64 v24; // r10
  unsigned int v25; // r8d
  __int64 *v26; // rax
  __int64 v27; // rdi
  __int64 **v28; // r11
  __int64 v29; // rax
  __int64 v30; // r8
  __int64 v31; // rax
  __int64 v32; // rax
  unsigned __int64 v33; // rdx
  __int64 v34; // r15
  __m128i *v35; // rsi
  __int32 v36; // ecx
  unsigned __int8 **v37; // r9
  unsigned __int64 v38; // rax
  int v39; // edx
  __int64 v40; // r15
  int v41; // eax
  unsigned __int64 v42; // r12
  __int64 v43; // r15
  __int64 v44; // r14
  __int64 v45; // r13
  __int64 v46; // r12
  __int64 v47; // rax
  int v48; // edx
  unsigned int v49; // ecx
  __int64 v50; // rsi
  unsigned int v51; // eax
  __int64 *v52; // rdi
  __int64 v53; // r10
  __int64 v54; // rax
  int v55; // edi
  int v56; // edx
  __int64 *v57; // rcx
  int v58; // eax
  int v59; // edi
  int v60; // edi
  int v61; // edx
  int v62; // esi
  int v63; // esi
  __int64 v64; // r8
  unsigned int v65; // eax
  __int64 v66; // rdx
  int v67; // r11d
  __int64 *v68; // r10
  int v69; // eax
  int v70; // esi
  __int64 v71; // r8
  int v72; // r11d
  unsigned int v73; // edx
  __int64 v74; // rax
  bool v75; // cc
  unsigned __int64 v76; // rax
  __int64 v77; // [rsp+0h] [rbp-A0h]
  __int64 v78; // [rsp+8h] [rbp-98h]
  __int64 **v79; // [rsp+10h] [rbp-90h]
  int v80; // [rsp+18h] [rbp-88h]
  __int64 v81; // [rsp+18h] [rbp-88h]
  __int64 v82; // [rsp+18h] [rbp-88h]
  unsigned __int64 v83; // [rsp+20h] [rbp-80h]
  unsigned __int64 v84; // [rsp+20h] [rbp-80h]
  int v85; // [rsp+2Ch] [rbp-74h]
  int v86; // [rsp+2Ch] [rbp-74h]
  __int64 v87; // [rsp+30h] [rbp-70h] BYREF
  __int64 v88; // [rsp+38h] [rbp-68h] BYREF
  __m128i v89; // [rsp+40h] [rbp-60h] BYREF
  __m128i v90[5]; // [rsp+50h] [rbp-50h] BYREF

  v10 = *(unsigned int *)(a1 + 88);
  v11 = *(_QWORD *)(a1 + 72);
  if ( (_DWORD)v10 )
  {
    a5 = (unsigned int)(v10 - 1);
    v12 = a5 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v13 = *(_QWORD *)(v11 + 16LL * v12);
    if ( a2 == v13 )
      return 0;
    v15 = 1;
    while ( v13 != -4096 )
    {
      a6 = (unsigned int)(v15 + 1);
      v12 = a5 & (v15 + v12);
      v13 = *(_QWORD *)(v11 + 16LL * v12);
      if ( a2 == v13 )
        return 0;
      ++v15;
    }
  }
  if ( a3 )
  {
    v87 = a3;
    v88 = a4;
    sub_3179990((__int64)&v89, a1 + 64, &v87, &v88);
    v17 = (unsigned __int8 *)a2;
    v18 = (__int64 *)a1;
    v19 = _mm_loadu_si128(v90);
    *(__m128i *)(a1 + 224) = _mm_loadu_si128(&v89);
    *(__m128i *)(a1 + 240) = v19;
    v20 = *(_BYTE *)a2;
    if ( *(_BYTE *)a2 == 32 )
    {
LABEL_8:
      v83 = sub_31794B0((__int64)v18, (__int64)v17);
      v85 = v22;
      goto LABEL_9;
    }
  }
  else
  {
    v16 = *(_QWORD *)(a1 + 64);
    *(_QWORD *)(a1 + 224) = a1 + 64;
    v18 = (__int64 *)a1;
    v47 = v11 + 16LL * (unsigned int)v10;
    *(_QWORD *)(a1 + 232) = v16;
    v17 = (unsigned __int8 *)a2;
    *(_QWORD *)(a1 + 240) = v47;
    *(_QWORD *)(a1 + 248) = v47;
    v20 = *(_BYTE *)a2;
    if ( *(_BYTE *)a2 == 32 )
      goto LABEL_8;
  }
  if ( v20 == 31 )
  {
    v83 = sub_3179820((__int64)v18, (__int64)v17, v16, v10, a5);
    v85 = v48;
  }
  else
  {
    v85 = 0;
    v83 = 0;
    a4 = sub_317A600(v18, v17, v16, v10, a5, a6);
    if ( !a4 )
      return 0;
  }
LABEL_9:
  v23 = *(_DWORD *)(a1 + 88);
  if ( !v23 )
  {
    ++*(_QWORD *)(a1 + 64);
    goto LABEL_62;
  }
  v24 = *(_QWORD *)(a1 + 72);
  v25 = (v23 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v26 = (__int64 *)(v24 + 16LL * v25);
  v27 = *v26;
  if ( a2 == *v26 )
    goto LABEL_11;
  v21 = 1;
  v57 = 0;
  while ( v27 != -4096 )
  {
    if ( v27 != -8192 || v57 )
      v26 = v57;
    v25 = (v23 - 1) & (v21 + v25);
    v27 = *(_QWORD *)(v24 + 16LL * v25);
    if ( a2 == v27 )
      goto LABEL_11;
    v21 = (unsigned int)(v21 + 1);
    v57 = v26;
    v26 = (__int64 *)(v24 + 16LL * v25);
  }
  if ( !v57 )
    v57 = v26;
  v58 = *(_DWORD *)(a1 + 80);
  ++*(_QWORD *)(a1 + 64);
  v59 = v58 + 1;
  if ( 4 * (v58 + 1) >= 3 * v23 )
  {
LABEL_62:
    sub_29D0890(a1 + 64, 2 * v23);
    v62 = *(_DWORD *)(a1 + 88);
    if ( v62 )
    {
      v63 = v62 - 1;
      v64 = *(_QWORD *)(a1 + 72);
      v65 = v63 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v59 = *(_DWORD *)(a1 + 80) + 1;
      v57 = (__int64 *)(v64 + 16LL * v65);
      v66 = *v57;
      if ( a2 == *v57 )
        goto LABEL_54;
      v67 = 1;
      v68 = 0;
      while ( v66 != -4096 )
      {
        if ( !v68 && v66 == -8192 )
          v68 = v57;
        v21 = (unsigned int)(v67 + 1);
        v65 = v63 & (v67 + v65);
        v57 = (__int64 *)(v64 + 16LL * v65);
        v66 = *v57;
        if ( a2 == *v57 )
          goto LABEL_54;
        ++v67;
      }
LABEL_66:
      if ( v68 )
        v57 = v68;
      goto LABEL_54;
    }
LABEL_93:
    ++*(_DWORD *)(a1 + 80);
    BUG();
  }
  if ( v23 - *(_DWORD *)(a1 + 84) - v59 <= v23 >> 3 )
  {
    sub_29D0890(a1 + 64, v23);
    v69 = *(_DWORD *)(a1 + 88);
    if ( v69 )
    {
      v70 = v69 - 1;
      v71 = *(_QWORD *)(a1 + 72);
      v68 = 0;
      v72 = 1;
      v73 = (v69 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v59 = *(_DWORD *)(a1 + 80) + 1;
      v57 = (__int64 *)(v71 + 16LL * v73);
      v74 = *v57;
      if ( a2 == *v57 )
        goto LABEL_54;
      while ( v74 != -4096 )
      {
        if ( !v68 && v74 == -8192 )
          v68 = v57;
        v21 = (unsigned int)(v72 + 1);
        v73 = v70 & (v72 + v73);
        v57 = (__int64 *)(v71 + 16LL * v73);
        v74 = *v57;
        if ( a2 == *v57 )
          goto LABEL_54;
        ++v72;
      }
      goto LABEL_66;
    }
    goto LABEL_93;
  }
LABEL_54:
  *(_DWORD *)(a1 + 80) = v59;
  if ( *v57 != -4096 )
    --*(_DWORD *)(a1 + 84);
  *v57 = a2;
  v57[1] = a4;
LABEL_11:
  v28 = *(__int64 ***)(a1 + 48);
  v29 = 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
  {
    v30 = *(_QWORD *)(a2 - 8);
    v31 = v30 + v29;
  }
  else
  {
    v30 = a2 - v29;
    v31 = a2;
  }
  v32 = v31 - v30;
  v89.m128i_i64[0] = (__int64)v90;
  v33 = v32 >> 5;
  v89.m128i_i64[1] = 0x400000000LL;
  v34 = v32 >> 5;
  if ( (unsigned __int64)v32 > 0x80 )
  {
    v77 = v32;
    v78 = v30;
    v79 = v28;
    v82 = v32 >> 5;
    sub_C8D5F0((__int64)&v89, v90, v33, 8u, v30, v21);
    v37 = (unsigned __int8 **)v89.m128i_i64[0];
    v36 = v89.m128i_i32[2];
    LODWORD(v33) = v82;
    v28 = v79;
    v30 = v78;
    v32 = v77;
    v35 = (__m128i *)(v89.m128i_i64[0] + 8LL * v89.m128i_u32[2]);
  }
  else
  {
    v35 = v90;
    v36 = 0;
    v37 = (unsigned __int8 **)v90;
  }
  if ( v32 > 0 )
  {
    v38 = 0;
    do
    {
      v35->m128i_i64[v38 / 8] = *(_QWORD *)(v30 + 4 * v38);
      v38 += 8LL;
      --v34;
    }
    while ( v34 );
    v37 = (unsigned __int8 **)v89.m128i_i64[0];
    v36 = v89.m128i_i32[2];
  }
  v89.m128i_i32[2] = v36 + v33;
  v40 = sub_DFCEF0(v28, (unsigned __int8 *)a2, v37, (unsigned int)(v36 + v33), 2);
  if ( (__m128i *)v89.m128i_i64[0] != v90 )
  {
    v80 = v39;
    _libc_free(v89.m128i_u64[0]);
    v39 = v80;
  }
  v41 = 1;
  if ( v39 != 1 )
    v41 = v85;
  v42 = v40 + v83;
  v86 = v41;
  if ( __OFADD__(v40, v83) )
  {
    v42 = 0x7FFFFFFFFFFFFFFFLL;
    if ( v40 <= 0 )
      v42 = 0x8000000000000000LL;
  }
  v43 = *(_QWORD *)(a2 + 16);
  if ( v43 )
  {
    v84 = v42;
    v81 = a4;
    v44 = a1;
    while ( 1 )
    {
      v45 = *(_QWORD *)(v43 + 24);
      if ( *(_BYTE *)v45 > 0x1Cu && v45 != a2 )
      {
        v46 = *(_QWORD *)(v45 + 40);
        if ( (unsigned __int8)sub_2A64220(*(__int64 **)(v44 + 56), v46) )
        {
          v49 = *(_DWORD *)(v44 + 120);
          v50 = *(_QWORD *)(v44 + 104);
          if ( !v49 )
            goto LABEL_39;
          v51 = (v49 - 1) & (((unsigned int)v46 >> 9) ^ ((unsigned int)v46 >> 4));
          v52 = (__int64 *)(v50 + 8LL * v51);
          v53 = *v52;
          if ( v46 != *v52 )
          {
            v60 = 1;
            while ( v53 != -4096 )
            {
              v61 = v60 + 1;
              v51 = (v49 - 1) & (v60 + v51);
              v52 = (__int64 *)(v50 + 8LL * v51);
              v53 = *v52;
              if ( v46 == *v52 )
                goto LABEL_38;
              v60 = v61;
            }
LABEL_39:
            v54 = sub_317A680(v44, v45, a2, v81);
            v55 = 1;
            if ( v56 != 1 )
              v55 = v86;
            v86 = v55;
            if ( __OFADD__(v54, v84) )
            {
              v75 = v54 <= 0;
              v76 = 0x8000000000000000LL;
              if ( !v75 )
                v76 = 0x7FFFFFFFFFFFFFFFLL;
              v84 = v76;
            }
            else
            {
              v84 += v54;
            }
            goto LABEL_29;
          }
LABEL_38:
          if ( v52 == (__int64 *)(v50 + 8LL * v49) )
            goto LABEL_39;
        }
      }
LABEL_29:
      v43 = *(_QWORD *)(v43 + 8);
      if ( !v43 )
        return v84;
    }
  }
  return v42;
}
