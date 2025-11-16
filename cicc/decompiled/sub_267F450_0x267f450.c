// Function: sub_267F450
// Address: 0x267f450
//
void __fastcall sub_267F450(_QWORD *a1, int a2, __int64 a3)
{
  __int64 v3; // r14
  __int64 v4; // rbx
  unsigned int v6; // esi
  __int64 v7; // rdi
  __int64 v8; // r9
  _QWORD *v9; // r13
  unsigned int v10; // r8d
  __int64 v11; // rcx
  __int64 v12; // r15
  __int64 v13; // rdx
  __int64 v14; // rax
  _QWORD *v15; // r13
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
  int v29; // eax
  int v30; // esi
  __int64 v31; // rdi
  unsigned int v32; // eax
  __int64 v33; // rdx
  _QWORD *v34; // rax
  int v35; // r11d
  int v36; // ecx
  int v37; // edx
  int v38; // esi
  __int64 v39; // rdi
  int v40; // r9d
  _QWORD *v41; // r8
  unsigned int v42; // eax
  __int64 v43; // rdx
  int v44; // r9d
  __m128i *v45; // [rsp+8h] [rbp-48h]
  unsigned __int64 v46[7]; // [rsp+18h] [rbp-38h] BYREF

  v3 = *(_QWORD *)(*a1 + 160LL * a2 + 3632);
  if ( !v3 )
    return;
  v4 = a1[1];
  v6 = *(_DWORD *)(v4 + 120);
  v7 = v4 + 96;
  if ( !v6 )
  {
    ++*(_QWORD *)(v4 + 96);
    goto LABEL_18;
  }
  v8 = *(_QWORD *)(v4 + 104);
  v9 = 0;
  v10 = (v6 - 1) & (((unsigned int)v3 >> 4) ^ ((unsigned int)v3 >> 9));
  v11 = 1;
  v12 = v8 + 56LL * v10;
  v13 = *(_QWORD *)v12;
  if ( v3 != *(_QWORD *)v12 )
  {
    while ( v13 != -4096 )
    {
      if ( v13 == -8192 && !v9 )
        v9 = (_QWORD *)v12;
      v35 = v11 + 1;
      v10 = (v6 - 1) & (v11 + v10);
      v11 = v10;
      v12 = v8 + 56LL * v10;
      v13 = *(_QWORD *)v12;
      if ( v3 == *(_QWORD *)v12 )
        goto LABEL_4;
      LODWORD(v11) = v35;
    }
    v36 = *(_DWORD *)(v4 + 112);
    if ( !v9 )
      v9 = (_QWORD *)v12;
    ++*(_QWORD *)(v4 + 96);
    v11 = (unsigned int)(v36 + 1);
    if ( 4 * (int)v11 < 3 * v6 )
    {
      if ( v6 - *(_DWORD *)(v4 + 116) - (unsigned int)v11 > v6 >> 3 )
        goto LABEL_20;
      sub_267F010(v7, v6);
      v37 = *(_DWORD *)(v4 + 120);
      if ( v37 )
      {
        v38 = v37 - 1;
        v39 = *(_QWORD *)(v4 + 104);
        v40 = 1;
        v41 = 0;
        v42 = (v37 - 1) & (((unsigned int)v3 >> 4) ^ ((unsigned int)v3 >> 9));
        v9 = (_QWORD *)(v39 + 56LL * v42);
        v43 = *v9;
        v11 = (unsigned int)(*(_DWORD *)(v4 + 112) + 1);
        if ( v3 != *v9 )
        {
          while ( v43 != -4096 )
          {
            if ( !v41 && v43 == -8192 )
              v41 = v9;
            v42 = v38 & (v40 + v42);
            v9 = (_QWORD *)(v39 + 56LL * v42);
            v43 = *v9;
            if ( v3 == *v9 )
              goto LABEL_20;
            ++v40;
          }
LABEL_36:
          if ( v41 )
            v9 = v41;
          goto LABEL_20;
        }
        goto LABEL_20;
      }
      goto LABEL_51;
    }
LABEL_18:
    sub_267F010(v7, 2 * v6);
    v29 = *(_DWORD *)(v4 + 120);
    if ( v29 )
    {
      v30 = v29 - 1;
      v31 = *(_QWORD *)(v4 + 104);
      v32 = (v29 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
      v9 = (_QWORD *)(v31 + 56LL * v32);
      v33 = *v9;
      v11 = (unsigned int)(*(_DWORD *)(v4 + 112) + 1);
      if ( v3 != *v9 )
      {
        v44 = 1;
        v41 = 0;
        while ( v33 != -4096 )
        {
          if ( !v41 && v33 == -8192 )
            v41 = v9;
          v32 = v30 & (v44 + v32);
          v9 = (_QWORD *)(v31 + 56LL * v32);
          v33 = *v9;
          if ( v3 == *v9 )
            goto LABEL_20;
          ++v44;
        }
        goto LABEL_36;
      }
LABEL_20:
      *(_DWORD *)(v4 + 112) = v11;
      if ( *v9 != -4096 )
        --*(_DWORD *)(v4 + 116);
      v34 = v9 + 3;
      *v9 = v3;
      v16 = 0;
      v15 = v9 + 1;
      *v15 = v34;
      v15[1] = 0x100000000LL;
      v14 = 0;
      goto LABEL_5;
    }
LABEL_51:
    ++*(_DWORD *)(v4 + 112);
    BUG();
  }
LABEL_4:
  v14 = *(unsigned int *)(v12 + 16);
  v15 = (_QWORD *)(v12 + 8);
  v16 = v14;
  if ( (unsigned int)v14 < *(_DWORD *)(v12 + 20) )
  {
LABEL_5:
    v17 = 32 * v14;
    v18 = *v15 + v17 == 0;
    v19 = *v15 + v17;
    v20 = v19;
    if ( !v18 )
    {
      *(_QWORD *)(v19 + 16) = 0;
      v21 = *(void (__fastcall **)(__int64, __int64, __int64, __int64))(a3 + 16);
      if ( v21 )
      {
        v21(v20, a3, 2, v11);
        *(_QWORD *)(v20 + 24) = *(_QWORD *)(a3 + 24);
        *(_QWORD *)(v20 + 16) = *(_QWORD *)(a3 + 16);
      }
      v16 = *((_DWORD *)v15 + 2);
    }
    *((_DWORD *)v15 + 2) = v16 + 1;
    return;
  }
  v22 = (__m128i *)sub_C8D7D0(v12 + 8, v12 + 24, 0, 0x20u, v46, v8);
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
  sub_26780F0(v12 + 8, v22);
  v27 = *(_QWORD *)(v12 + 8);
  v28 = v46[0];
  if ( v12 + 24 != v27 )
    _libc_free(v27);
  ++*(_DWORD *)(v12 + 16);
  *(_QWORD *)(v12 + 8) = v22;
  *(_DWORD *)(v12 + 20) = v28;
}
