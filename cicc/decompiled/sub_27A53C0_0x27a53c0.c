// Function: sub_27A53C0
// Address: 0x27a53c0
//
__int64 __fastcall sub_27A53C0(__int64 a1, __int64 a2)
{
  __int64 v4; // rdi
  unsigned int v5; // esi
  int v6; // r11d
  __int64 v7; // rcx
  __int64 *v8; // r14
  unsigned int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // r9
  __int64 result; // rax
  int v13; // eax
  int v14; // edx
  __int64 v15; // rax
  unsigned __int64 v16; // rax
  __int64 v17; // rdx
  unsigned __int64 v18; // rdi
  int v19; // eax
  unsigned __int8 *v20; // rdi
  int v21; // eax
  int v22; // ecx
  __int64 v23; // rdi
  unsigned int v24; // eax
  __int64 v25; // rsi
  int v26; // r9d
  __int64 *v27; // r8
  int v28; // eax
  int v29; // eax
  __int64 v30; // rsi
  __int64 *v31; // rdi
  unsigned int v32; // r12d
  int v33; // r8d
  __int64 v34; // rcx

  v4 = a1 + 296;
  v5 = *(_DWORD *)(a1 + 320);
  if ( !v5 )
  {
    ++*(_QWORD *)(a1 + 296);
    goto LABEL_29;
  }
  v6 = 1;
  v7 = *(_QWORD *)(a1 + 304);
  v8 = 0;
  v9 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v10 = (__int64 *)(v7 + 16LL * v9);
  v11 = *v10;
  if ( *v10 == a2 )
    return *((unsigned __int8 *)v10 + 8);
  while ( v11 != -4096 )
  {
    if ( v11 == -8192 && !v8 )
      v8 = v10;
    v9 = (v5 - 1) & (v6 + v9);
    v10 = (__int64 *)(v7 + 16LL * v9);
    v11 = *v10;
    if ( *v10 == a2 )
      return *((unsigned __int8 *)v10 + 8);
    ++v6;
  }
  if ( !v8 )
    v8 = v10;
  v13 = *(_DWORD *)(a1 + 312);
  ++*(_QWORD *)(a1 + 296);
  v14 = v13 + 1;
  if ( 4 * (v13 + 1) >= 3 * v5 )
  {
LABEL_29:
    sub_11F63E0(v4, 2 * v5);
    v21 = *(_DWORD *)(a1 + 320);
    if ( v21 )
    {
      v22 = v21 - 1;
      v23 = *(_QWORD *)(a1 + 304);
      v24 = (v21 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v14 = *(_DWORD *)(a1 + 312) + 1;
      v8 = (__int64 *)(v23 + 16LL * v24);
      v25 = *v8;
      if ( *v8 != a2 )
      {
        v26 = 1;
        v27 = 0;
        while ( v25 != -4096 )
        {
          if ( !v27 && v25 == -8192 )
            v27 = v8;
          v24 = v22 & (v26 + v24);
          v8 = (__int64 *)(v23 + 16LL * v24);
          v25 = *v8;
          if ( *v8 == a2 )
            goto LABEL_15;
          ++v26;
        }
        if ( v27 )
          v8 = v27;
      }
      goto LABEL_15;
    }
    goto LABEL_53;
  }
  if ( v5 - *(_DWORD *)(a1 + 316) - v14 <= v5 >> 3 )
  {
    sub_11F63E0(v4, v5);
    v28 = *(_DWORD *)(a1 + 320);
    if ( v28 )
    {
      v29 = v28 - 1;
      v30 = *(_QWORD *)(a1 + 304);
      v31 = 0;
      v32 = v29 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v33 = 1;
      v14 = *(_DWORD *)(a1 + 312) + 1;
      v8 = (__int64 *)(v30 + 16LL * v32);
      v34 = *v8;
      if ( *v8 != a2 )
      {
        while ( v34 != -4096 )
        {
          if ( v34 == -8192 && !v31 )
            v31 = v8;
          v32 = v29 & (v33 + v32);
          v8 = (__int64 *)(v30 + 16LL * v32);
          v34 = *v8;
          if ( *v8 == a2 )
            goto LABEL_15;
          ++v33;
        }
        if ( v31 )
          v8 = v31;
      }
      goto LABEL_15;
    }
LABEL_53:
    ++*(_DWORD *)(a1 + 312);
    BUG();
  }
LABEL_15:
  *(_DWORD *)(a1 + 312) = v14;
  if ( *v8 != -4096 )
    --*(_DWORD *)(a1 + 316);
  *v8 = a2;
  *((_BYTE *)v8 + 8) = 0;
  v15 = sub_AA4FF0(a2);
  if ( !v15 )
    BUG();
  v16 = (unsigned int)*(unsigned __int8 *)(v15 - 24) - 39;
  if ( (unsigned int)v16 <= 0x38 && (v17 = 0x100060000000001LL, _bittest64(&v17, v16))
    || (*(_WORD *)(a2 + 2) & 0x7FFF) != 0 )
  {
    *((_BYTE *)v8 + 8) = 1;
    return 1;
  }
  else
  {
    v18 = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v18 == a2 + 48 )
    {
      v20 = 0;
    }
    else
    {
      if ( !v18 )
        BUG();
      v19 = *(unsigned __int8 *)(v18 - 24);
      v20 = (unsigned __int8 *)(v18 - 24);
      if ( (unsigned int)(v19 - 30) >= 0xB )
        v20 = 0;
    }
    result = sub_B46790(v20, 0);
    if ( (_BYTE)result )
      *((_BYTE *)v8 + 8) = 1;
  }
  return result;
}
