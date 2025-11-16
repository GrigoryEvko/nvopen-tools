// Function: sub_1E0BEF0
// Address: 0x1e0bef0
//
__int64 __fastcall sub_1E0BEF0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v8; // rbx
  __int64 v9; // rdx
  unsigned __int64 v10; // r13
  __int64 v11; // r12
  __int64 v12; // rax
  __int64 v13; // rbx
  __int64 v14; // rdx
  __int64 v15; // rcx
  int v16; // r8d
  int v17; // r9d
  __int64 v18; // rdx
  __int64 v19; // rcx
  int v20; // r8d
  int v21; // r9d
  __int64 v22; // rdi
  __int64 v23; // rsi
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rdi
  unsigned __int64 v28; // rdi
  unsigned __int64 v29; // rdi
  unsigned __int64 v30; // rdi

  v8 = *(_QWORD *)(a1 + 8);
  if ( a2 + 120 != v8 )
  {
    v9 = v8 - (a2 + 120);
    v10 = 0xEEEEEEEEEEEEEEEFLL * (v9 >> 3);
    if ( v9 > 0 )
    {
      v11 = a2 + 8;
      do
      {
        v12 = *(_QWORD *)(v11 + 112);
        v13 = v11;
        v11 += 120;
        *(_QWORD *)(v11 - 128) = v12;
        sub_1E09880(v13, (char **)v11, v9, a4, a5, a6);
        sub_1E09880(v13 + 24, (char **)(v13 + 144), v14, v15, v16, v17);
        sub_1E096E0(v13 + 48, (char **)(v13 + 168), v18, v19, v20, v21);
        v22 = *(_QWORD *)(v13 + 88);
        v23 = *(_QWORD *)(v13 + 104);
        *(_QWORD *)(v13 + 80) = *(_QWORD *)(v13 + 200);
        v24 = *(_QWORD *)(v13 + 208);
        *(_QWORD *)(v13 + 208) = 0;
        *(_QWORD *)(v13 + 88) = v24;
        v25 = *(_QWORD *)(v13 + 216);
        *(_QWORD *)(v13 + 216) = 0;
        *(_QWORD *)(v13 + 96) = v25;
        v26 = *(_QWORD *)(v13 + 224);
        *(_QWORD *)(v13 + 224) = 0;
        *(_QWORD *)(v13 + 104) = v26;
        if ( v22 )
          j_j___libc_free_0(v22, v23 - v22);
        --v10;
      }
      while ( v10 );
      v8 = *(_QWORD *)(a1 + 8);
    }
  }
  *(_QWORD *)(a1 + 8) = v8 - 120;
  v27 = *(_QWORD *)(v8 - 24);
  if ( v27 )
    j_j___libc_free_0(v27, *(_QWORD *)(v8 - 8) - v27);
  v28 = *(_QWORD *)(v8 - 64);
  if ( v28 != v8 - 48 )
    _libc_free(v28);
  v29 = *(_QWORD *)(v8 - 88);
  if ( v29 != v8 - 72 )
    _libc_free(v29);
  v30 = *(_QWORD *)(v8 - 112);
  if ( v30 != v8 - 96 )
    _libc_free(v30);
  return a2;
}
