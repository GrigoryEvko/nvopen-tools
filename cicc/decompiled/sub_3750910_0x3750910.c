// Function: sub_3750910
// Address: 0x3750910
//
__int64 __fastcall sub_3750910(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r9
  unsigned int v7; // esi
  __int64 v8; // r8
  __int64 v9; // rcx
  int v10; // r11d
  __int64 v11; // r10
  unsigned int v12; // edx
  __int64 v13; // r12
  __int64 v14; // rax
  int v16; // eax
  int v17; // edx
  int v18; // eax
  int v19; // ecx
  __int64 v20; // rsi
  unsigned int v21; // eax
  __int64 v22; // rdi
  int v23; // r10d
  int v24; // eax
  int v25; // eax
  __int64 v26; // rsi
  unsigned int v27; // r15d
  __int64 v28; // rdi
  __int64 v29; // rcx
  __int64 v30; // [rsp+8h] [rbp-38h]
  __int64 v31; // [rsp+8h] [rbp-38h]

  v3 = a1 + 184;
  v7 = *(_DWORD *)(a1 + 208);
  v8 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 32LL);
  if ( !v7 )
  {
    ++*(_QWORD *)(a1 + 184);
    goto LABEL_18;
  }
  v9 = *(_QWORD *)(a1 + 192);
  v10 = 1;
  v11 = 0;
  v12 = (v7 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v13 = v9 + 16LL * v12;
  v14 = *(_QWORD *)v13;
  if ( a2 == *(_QWORD *)v13 )
    return *(unsigned int *)(v13 + 8);
  while ( v14 != -4096 )
  {
    if ( v14 == -8192 && !v11 )
      v11 = v13;
    v12 = (v7 - 1) & (v10 + v12);
    v13 = v9 + 16LL * v12;
    v14 = *(_QWORD *)v13;
    if ( a2 == *(_QWORD *)v13 )
      return *(unsigned int *)(v13 + 8);
    ++v10;
  }
  v16 = *(_DWORD *)(a1 + 200);
  if ( v11 )
    v13 = v11;
  ++*(_QWORD *)(a1 + 184);
  v17 = v16 + 1;
  if ( 4 * (v16 + 1) >= 3 * v7 )
  {
LABEL_18:
    v30 = v8;
    sub_3384500(v3, 2 * v7);
    v18 = *(_DWORD *)(a1 + 208);
    if ( v18 )
    {
      v19 = v18 - 1;
      v8 = v30;
      v20 = *(_QWORD *)(a1 + 192);
      v21 = (v18 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v17 = *(_DWORD *)(a1 + 200) + 1;
      v13 = v20 + 16LL * v21;
      v22 = *(_QWORD *)v13;
      if ( a2 != *(_QWORD *)v13 )
      {
        v23 = 1;
        v3 = 0;
        while ( v22 != -4096 )
        {
          if ( !v3 && v22 == -8192 )
            v3 = v13;
          v21 = v19 & (v23 + v21);
          v13 = v20 + 16LL * v21;
          v22 = *(_QWORD *)v13;
          if ( a2 == *(_QWORD *)v13 )
            goto LABEL_14;
          ++v23;
        }
        if ( v3 )
          v13 = v3;
      }
      goto LABEL_14;
    }
    goto LABEL_41;
  }
  if ( v7 - *(_DWORD *)(a1 + 204) - v17 <= v7 >> 3 )
  {
    v31 = v8;
    sub_3384500(v3, v7);
    v24 = *(_DWORD *)(a1 + 208);
    if ( v24 )
    {
      v25 = v24 - 1;
      v26 = *(_QWORD *)(a1 + 192);
      v3 = 1;
      v27 = v25 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v8 = v31;
      v17 = *(_DWORD *)(a1 + 200) + 1;
      v28 = 0;
      v13 = v26 + 16LL * v27;
      v29 = *(_QWORD *)v13;
      if ( a2 != *(_QWORD *)v13 )
      {
        while ( v29 != -4096 )
        {
          if ( !v28 && v29 == -8192 )
            v28 = v13;
          v27 = v25 & (v3 + v27);
          v13 = v26 + 16LL * v27;
          v29 = *(_QWORD *)v13;
          if ( a2 == *(_QWORD *)v13 )
            goto LABEL_14;
          v3 = (unsigned int)(v3 + 1);
        }
        if ( v28 )
          v13 = v28;
      }
      goto LABEL_14;
    }
LABEL_41:
    ++*(_DWORD *)(a1 + 200);
    BUG();
  }
LABEL_14:
  *(_DWORD *)(a1 + 200) = v17;
  if ( *(_QWORD *)v13 != -4096 )
    --*(_DWORD *)(a1 + 204);
  *(_QWORD *)v13 = a2;
  *(_DWORD *)(v13 + 8) = 0;
  *(_DWORD *)(v13 + 8) = sub_2EC06C0(v8, a3, byte_3F871B3, 0, v8, v3);
  return *(unsigned int *)(v13 + 8);
}
