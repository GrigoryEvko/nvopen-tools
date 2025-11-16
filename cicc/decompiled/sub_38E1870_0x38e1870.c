// Function: sub_38E1870
// Address: 0x38e1870
//
__int64 __fastcall sub_38E1870(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 v3; // r9
  unsigned int v6; // esi
  int v7; // r15d
  __int64 v8; // rdi
  int v9; // r14d
  unsigned int v10; // ecx
  __int64 result; // rax
  __int64 *v12; // rdx
  int v13; // r11d
  __int64 v14; // r10
  int v15; // ecx
  int v16; // eax
  int v17; // esi
  __int64 v18; // rdi
  unsigned int v19; // edx
  __int64 *v20; // r8
  int v21; // r10d
  __int64 v22; // r9
  int v23; // eax
  int v24; // edx
  __int64 v25; // rdi
  int v26; // r9d
  unsigned int v27; // r13d
  __int64 v28; // r8
  __int64 *v29; // rsi

  v3 = a1 + 80;
  *a2 = *a2 & 7 | a3;
  v6 = *(_DWORD *)(a1 + 104);
  v7 = *(_DWORD *)(a1 + 96);
  if ( !v6 )
  {
    ++*(_QWORD *)(a1 + 80);
    goto LABEL_14;
  }
  v8 = *(_QWORD *)(a1 + 88);
  v9 = v7 + 1;
  v10 = (v6 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  result = v8 + 16LL * v10;
  v12 = *(__int64 **)result;
  if ( a2 == *(__int64 **)result )
    goto LABEL_3;
  v13 = 1;
  v14 = 0;
  while ( v12 != (__int64 *)-8LL )
  {
    if ( !v14 && v12 == (__int64 *)-16LL )
      v14 = result;
    v10 = (v6 - 1) & (v13 + v10);
    result = v8 + 16LL * v10;
    v12 = *(__int64 **)result;
    if ( *(__int64 **)result == a2 )
      goto LABEL_3;
    ++v13;
  }
  if ( v14 )
    result = v14;
  ++*(_QWORD *)(a1 + 80);
  if ( 4 * v9 >= 3 * v6 )
  {
LABEL_14:
    sub_211A5E0(v3, 2 * v6);
    v16 = *(_DWORD *)(a1 + 104);
    if ( v16 )
    {
      v17 = v16 - 1;
      v18 = *(_QWORD *)(a1 + 88);
      v9 = v7 + 1;
      v19 = (v16 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v15 = *(_DWORD *)(a1 + 96) + 1;
      result = v18 + 16LL * v19;
      v20 = *(__int64 **)result;
      if ( *(__int64 **)result != a2 )
      {
        v21 = 1;
        v22 = 0;
        while ( v20 != (__int64 *)-8LL )
        {
          if ( !v22 && v20 == (__int64 *)-16LL )
            v22 = result;
          v19 = v17 & (v21 + v19);
          result = v18 + 16LL * v19;
          v20 = *(__int64 **)result;
          if ( *(__int64 **)result == a2 )
            goto LABEL_10;
          ++v21;
        }
        if ( v22 )
          result = v22;
      }
      goto LABEL_10;
    }
    goto LABEL_42;
  }
  v15 = v7 + 1;
  if ( v6 - *(_DWORD *)(a1 + 100) - v9 <= v6 >> 3 )
  {
    sub_211A5E0(v3, v6);
    v23 = *(_DWORD *)(a1 + 104);
    if ( v23 )
    {
      v24 = v23 - 1;
      v25 = *(_QWORD *)(a1 + 88);
      v26 = 1;
      v27 = (v23 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v28 = 0;
      v15 = *(_DWORD *)(a1 + 96) + 1;
      result = v25 + 16LL * v27;
      v29 = *(__int64 **)result;
      if ( *(__int64 **)result != a2 )
      {
        while ( v29 != (__int64 *)-8LL )
        {
          if ( v29 == (__int64 *)-16LL && !v28 )
            v28 = result;
          v27 = v24 & (v26 + v27);
          result = v25 + 16LL * v27;
          v29 = *(__int64 **)result;
          if ( *(__int64 **)result == a2 )
            goto LABEL_10;
          ++v26;
        }
        if ( v28 )
          result = v28;
      }
      goto LABEL_10;
    }
LABEL_42:
    ++*(_DWORD *)(a1 + 96);
    BUG();
  }
LABEL_10:
  *(_DWORD *)(a1 + 96) = v15;
  if ( *(_QWORD *)result != -8 )
    --*(_DWORD *)(a1 + 100);
  *(_QWORD *)result = a2;
  *(_DWORD *)(result + 8) = 0;
LABEL_3:
  *(_DWORD *)(result + 8) = v9;
  return result;
}
