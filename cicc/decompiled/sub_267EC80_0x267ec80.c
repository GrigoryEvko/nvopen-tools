// Function: sub_267EC80
// Address: 0x267ec80
//
void __fastcall sub_267EC80(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rdi
  unsigned int v7; // esi
  __int64 v8; // r9
  _QWORD *v9; // r12
  unsigned int v10; // r8d
  __int64 v11; // rcx
  __int64 v12; // r15
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 *v15; // r12
  int v16; // edx
  __int64 v17; // rax
  bool v18; // zf
  __int64 v19; // rax
  __int64 v20; // rbx
  void (__fastcall *v21)(__int64, __int64, __int64, __int64); // rax
  __m128i *v22; // r14
  __int64 v23; // rax
  __m128i *v24; // rax
  __m128i *v25; // rcx
  void (__fastcall *v26)(__m128i *, __int64, __int64); // rax
  unsigned __int64 v27; // rdi
  int v28; // r12d
  int v29; // r11d
  int v30; // ecx
  _QWORD *v31; // rax
  int v32; // eax
  int v33; // esi
  __int64 v34; // rdi
  unsigned int v35; // eax
  __int64 v36; // rdx
  int v37; // r9d
  _QWORD *v38; // r8
  int v39; // edx
  int v40; // edx
  __int64 v41; // rdi
  int v42; // r9d
  unsigned int v43; // eax
  __int64 v44; // rsi
  __m128i *v45; // [rsp+8h] [rbp-48h]
  unsigned int v46; // [rsp+8h] [rbp-48h]
  unsigned __int64 v47[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = a1 + 64;
  v7 = *(_DWORD *)(a1 + 88);
  if ( !v7 )
  {
    ++*(_QWORD *)(a1 + 64);
    goto LABEL_30;
  }
  v8 = *(_QWORD *)(a1 + 72);
  v9 = 0;
  v10 = (v7 - 1) & (((unsigned int)a2 >> 4) ^ ((unsigned int)a2 >> 9));
  v11 = 1;
  v12 = v8 + 56LL * v10;
  v13 = *(_QWORD *)v12;
  if ( a2 != *(_QWORD *)v12 )
  {
    while ( v13 != -4096 )
    {
      if ( v13 == -8192 && !v9 )
        v9 = (_QWORD *)v12;
      v29 = v11 + 1;
      v10 = (v7 - 1) & (v11 + v10);
      v11 = v10;
      v12 = v8 + 56LL * v10;
      v13 = *(_QWORD *)v12;
      if ( a2 == *(_QWORD *)v12 )
        goto LABEL_3;
      LODWORD(v11) = v29;
    }
    v30 = *(_DWORD *)(a1 + 80);
    if ( !v9 )
      v9 = (_QWORD *)v12;
    ++*(_QWORD *)(a1 + 64);
    v11 = (unsigned int)(v30 + 1);
    if ( 4 * (int)v11 < 3 * v7 )
    {
      if ( v7 - *(_DWORD *)(a1 + 84) - (unsigned int)v11 > v7 >> 3 )
        goto LABEL_25;
      v46 = ((unsigned int)a2 >> 4) ^ ((unsigned int)a2 >> 9);
      sub_267E840(v6, v7);
      v39 = *(_DWORD *)(a1 + 88);
      if ( v39 )
      {
        v40 = v39 - 1;
        v41 = *(_QWORD *)(a1 + 72);
        v42 = 1;
        v38 = 0;
        v43 = v40 & v46;
        v9 = (_QWORD *)(v41 + 56LL * (v40 & v46));
        v44 = *v9;
        v11 = (unsigned int)(*(_DWORD *)(a1 + 80) + 1);
        if ( a2 == *v9 )
          goto LABEL_25;
        while ( v44 != -4096 )
        {
          if ( !v38 && v44 == -8192 )
            v38 = v9;
          v43 = v40 & (v42 + v43);
          v9 = (_QWORD *)(v41 + 56LL * v43);
          v44 = *v9;
          if ( a2 == *v9 )
            goto LABEL_25;
          ++v42;
        }
LABEL_34:
        if ( v38 )
          v9 = v38;
LABEL_25:
        *(_DWORD *)(a1 + 80) = v11;
        if ( *v9 != -4096 )
          --*(_DWORD *)(a1 + 84);
        v31 = v9 + 3;
        *v9 = a2;
        v15 = v9 + 1;
        v16 = 0;
        *v15 = (__int64)v31;
        v15[1] = 0x100000000LL;
        v19 = *v15;
        v20 = *v15;
        if ( !*v15 )
          goto LABEL_8;
        goto LABEL_5;
      }
      goto LABEL_50;
    }
LABEL_30:
    sub_267E840(v6, 2 * v7);
    v32 = *(_DWORD *)(a1 + 88);
    if ( v32 )
    {
      v33 = v32 - 1;
      v34 = *(_QWORD *)(a1 + 72);
      v35 = (v32 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v9 = (_QWORD *)(v34 + 56LL * v35);
      v36 = *v9;
      v11 = (unsigned int)(*(_DWORD *)(a1 + 80) + 1);
      if ( a2 == *v9 )
        goto LABEL_25;
      v37 = 1;
      v38 = 0;
      while ( v36 != -4096 )
      {
        if ( !v38 && v36 == -8192 )
          v38 = v9;
        v35 = v33 & (v37 + v35);
        v9 = (_QWORD *)(v34 + 56LL * v35);
        v36 = *v9;
        if ( a2 == *v9 )
          goto LABEL_25;
        ++v37;
      }
      goto LABEL_34;
    }
LABEL_50:
    ++*(_DWORD *)(a1 + 80);
    BUG();
  }
LABEL_3:
  v14 = *(unsigned int *)(v12 + 16);
  v15 = (__int64 *)(v12 + 8);
  v16 = v14;
  if ( *(_DWORD *)(v12 + 20) > (unsigned int)v14 )
  {
    v17 = 32 * v14;
    v18 = *v15 + v17 == 0;
    v19 = *v15 + v17;
    v20 = v19;
    if ( v18 )
    {
LABEL_8:
      *((_DWORD *)v15 + 2) = v16 + 1;
      return;
    }
LABEL_5:
    *(_QWORD *)(v19 + 16) = 0;
    v21 = *(void (__fastcall **)(__int64, __int64, __int64, __int64))(a3 + 16);
    if ( v21 )
    {
      v21(v20, a3, 2, v11);
      *(_QWORD *)(v20 + 24) = *(_QWORD *)(a3 + 24);
      *(_QWORD *)(v20 + 16) = *(_QWORD *)(a3 + 16);
    }
    v16 = *((_DWORD *)v15 + 2);
    goto LABEL_8;
  }
  v22 = (__m128i *)sub_C8D7D0(v12 + 8, v12 + 24, 0, 0x20u, v47, v8);
  v23 = 2LL * *(unsigned int *)(v12 + 16);
  v18 = &v22[v23] == 0;
  v24 = &v22[v23];
  v25 = v24;
  if ( !v18 )
  {
    v24[1].m128i_i64[0] = 0;
    v26 = *(void (__fastcall **)(__m128i *, __int64, __int64))(a3 + 16);
    if ( v26 )
    {
      v45 = v25;
      v26(v25, a3, 2);
      v45[1].m128i_i64[1] = *(_QWORD *)(a3 + 24);
      v45[1].m128i_i64[0] = *(_QWORD *)(a3 + 16);
    }
  }
  sub_2678040(v12 + 8, v22);
  v27 = *(_QWORD *)(v12 + 8);
  v28 = v47[0];
  if ( v12 + 24 != v27 )
    _libc_free(v27);
  ++*(_DWORD *)(v12 + 16);
  *(_QWORD *)(v12 + 8) = v22;
  *(_DWORD *)(v12 + 20) = v28;
}
