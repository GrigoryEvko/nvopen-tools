// Function: sub_1860630
// Address: 0x1860630
//
__int64 __fastcall sub_1860630(__int64 a1, unsigned __int32 a2, __int64 a3, __int64 a4)
{
  unsigned int v7; // esi
  __int64 v8; // rcx
  unsigned int v9; // r14d
  unsigned int v10; // edx
  __int64 *v11; // rbx
  __int64 v12; // rax
  __int64 v13; // rcx
  __int64 v14; // r14
  unsigned __int64 v15; // rdx
  unsigned __int64 v16; // r8
  __int64 v17; // rax
  __int64 v18; // r14
  unsigned int v20; // esi
  __int64 v21; // rcx
  __int64 v22; // rdx
  int v23; // r15d
  __int64 v24; // rax
  __m128i *v25; // rsi
  __int64 v26; // rax
  __int64 v27; // r15
  __int64 v28; // rdx
  _QWORD *v29; // rax
  int v30; // r9d
  __int64 *v31; // r8
  int v32; // eax
  int v33; // edx
  unsigned __int64 v34; // rdi
  __int64 v35; // r9
  int v36; // eax
  int v37; // ecx
  __int64 v38; // rsi
  unsigned int v39; // eax
  __int64 v40; // rdi
  int v41; // r9d
  __int64 *v42; // r8
  int v43; // eax
  int v44; // eax
  __int64 v45; // rsi
  int v46; // r8d
  unsigned int v47; // r14d
  __int64 *v48; // rdi
  __int64 v49; // rcx
  __int64 v50; // rdx
  __int64 v51; // [rsp+8h] [rbp-B8h]
  __int64 v52; // [rsp+8h] [rbp-B8h]
  __int64 v54; // [rsp+18h] [rbp-A8h]
  const char *v55; // [rsp+20h] [rbp-A0h] BYREF
  __int64 v56; // [rsp+28h] [rbp-98h]
  const char **v57; // [rsp+30h] [rbp-90h] BYREF
  char *v58; // [rsp+38h] [rbp-88h]
  __int64 v59; // [rsp+40h] [rbp-80h]
  __int64 v60; // [rsp+50h] [rbp-70h]
  __int16 v61; // [rsp+60h] [rbp-60h]
  __m128i v62; // [rsp+70h] [rbp-50h] BYREF
  __int64 v63; // [rsp+80h] [rbp-40h]

  v7 = *(_DWORD *)(a3 + 24);
  if ( !v7 )
  {
    ++*(_QWORD *)a3;
    goto LABEL_29;
  }
  v8 = *(_QWORD *)(a3 + 8);
  v9 = ((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4);
  v10 = (v7 - 1) & v9;
  v11 = (__int64 *)(v8 + 32LL * v10);
  v12 = *v11;
  if ( *v11 != a1 )
  {
    v30 = 1;
    v31 = 0;
    while ( v12 != -8 )
    {
      if ( v12 == -16 && !v31 )
        v31 = v11;
      v10 = (v7 - 1) & (v30 + v10);
      v11 = (__int64 *)(v8 + 32LL * v10);
      v12 = *v11;
      if ( *v11 == a1 )
        goto LABEL_3;
      ++v30;
    }
    v32 = *(_DWORD *)(a3 + 16);
    if ( v31 )
      v11 = v31;
    ++*(_QWORD *)a3;
    v33 = v32 + 1;
    if ( 4 * (v32 + 1) < 3 * v7 )
    {
      if ( v7 - *(_DWORD *)(a3 + 20) - v33 > v7 >> 3 )
        goto LABEL_22;
      sub_1860410(a3, v7);
      v43 = *(_DWORD *)(a3 + 24);
      if ( v43 )
      {
        v44 = v43 - 1;
        v45 = *(_QWORD *)(a3 + 8);
        v46 = 1;
        v47 = v44 & v9;
        v33 = *(_DWORD *)(a3 + 16) + 1;
        v48 = 0;
        v11 = (__int64 *)(v45 + 32LL * v47);
        v49 = *v11;
        if ( *v11 != a1 )
        {
          while ( v49 != -8 )
          {
            if ( !v48 && v49 == -16 )
              v48 = v11;
            v47 = v44 & (v46 + v47);
            v11 = (__int64 *)(v45 + 32LL * v47);
            v49 = *v11;
            if ( *v11 == a1 )
              goto LABEL_22;
            ++v46;
          }
          if ( v48 )
            v11 = v48;
        }
        goto LABEL_22;
      }
      goto LABEL_61;
    }
LABEL_29:
    sub_1860410(a3, 2 * v7);
    v36 = *(_DWORD *)(a3 + 24);
    if ( v36 )
    {
      v37 = v36 - 1;
      v38 = *(_QWORD *)(a3 + 8);
      v39 = (v36 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v33 = *(_DWORD *)(a3 + 16) + 1;
      v11 = (__int64 *)(v38 + 32LL * v39);
      v40 = *v11;
      if ( *v11 != a1 )
      {
        v41 = 1;
        v42 = 0;
        while ( v40 != -8 )
        {
          if ( !v42 && v40 == -16 )
            v42 = v11;
          v39 = v37 & (v41 + v39);
          v11 = (__int64 *)(v38 + 32LL * v39);
          v40 = *v11;
          if ( *v11 == a1 )
            goto LABEL_22;
          ++v41;
        }
        if ( v42 )
          v11 = v42;
      }
LABEL_22:
      *(_DWORD *)(a3 + 16) = v33;
      if ( *v11 != -8 )
        --*(_DWORD *)(a3 + 20);
      v34 = a2 + 1;
      *v11 = a1;
      v35 = (__int64)(v11 + 1);
      v14 = a2;
      v11[1] = 0;
      v16 = 0;
      v11[2] = 0;
      v11[3] = 0;
      if ( a2 == -1 )
      {
        v13 = 0;
        goto LABEL_4;
      }
LABEL_25:
      sub_185FB80(v35, v34 - v16);
      v13 = v11[1];
      goto LABEL_4;
    }
LABEL_61:
    ++*(_DWORD *)(a3 + 16);
    BUG();
  }
LABEL_3:
  v13 = v11[1];
  v14 = a2;
  v15 = (v11[2] - v13) >> 3;
  v16 = v15;
  if ( a2 < v15 )
    goto LABEL_4;
  v34 = a2 + 1;
  v35 = (__int64)(v11 + 1);
  if ( v34 > v15 )
    goto LABEL_25;
  if ( v34 < v15 )
  {
    v50 = v13 + 8 * v34;
    if ( v11[2] != v50 )
      v11[2] = v50;
  }
LABEL_4:
  v17 = 8 * v14;
  v18 = *(_QWORD *)(v13 + 8 * v14);
  v54 = v17;
  if ( !v18 )
  {
    if ( *(_BYTE *)(a1 + 16) == 54 )
    {
      v26 = sub_1860630(*(_QWORD *)(a1 - 24), a2, a3, a4, v16);
      LODWORD(v60) = a2;
      v61 = 265;
      v27 = v26;
      v55 = sub_1649960(a1);
      v57 = &v55;
      v58 = ".f";
      v56 = v28;
      LOWORD(v59) = 773;
      v62.m128i_i64[1] = v60;
      v62.m128i_i64[0] = (__int64)&v57;
      LOWORD(v63) = 2306;
      v29 = sub_1648A60(64, 1u);
      v18 = (__int64)v29;
      if ( v29 )
        sub_15F90E0((__int64)v29, v27, (__int64)&v62, a1);
    }
    else
    {
      v20 = *(_DWORD *)(*(_QWORD *)a1 + 8LL);
      v21 = *(_QWORD *)(*(_QWORD *)a1 + 24LL);
      v61 = 265;
      LODWORD(v60) = a2;
      v51 = v21;
      v55 = sub_1649960(a1);
      v57 = &v55;
      v58 = ".f";
      LOWORD(v59) = 773;
      v56 = v22;
      v62.m128i_i64[1] = v60;
      v62.m128i_i64[0] = (__int64)&v57;
      LOWORD(v63) = 2306;
      v23 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
      v52 = sub_1646BA0(*(__int64 **)(*(_QWORD *)(v51 + 16) + v54), v20 >> 8);
      v24 = sub_1648B60(64);
      v18 = v24;
      if ( v24 )
      {
        sub_15F1EA0(v24, v52, 53, 0, 0, a1);
        *(_DWORD *)(v18 + 56) = v23;
        sub_164B780(v18, v62.m128i_i64);
        sub_1648880(v18, *(_DWORD *)(v18 + 56), 1);
      }
      v62.m128i_i64[0] = a1;
      v62.m128i_i32[2] = a2;
      v25 = *(__m128i **)(a4 + 8);
      if ( v25 == *(__m128i **)(a4 + 16) )
      {
        sub_1860290((const __m128i **)a4, v25, &v62);
      }
      else
      {
        if ( v25 )
        {
          *v25 = _mm_loadu_si128(&v62);
          v25 = *(__m128i **)(a4 + 8);
        }
        *(_QWORD *)(a4 + 8) = v25 + 1;
      }
    }
    *(_QWORD *)(v11[1] + v54) = v18;
  }
  return v18;
}
