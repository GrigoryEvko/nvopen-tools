// Function: sub_12DE0B0
// Address: 0x12de0b0
//
__int64 __fastcall sub_12DE0B0(__int64 a1, __int64 a2, unsigned __int8 a3, char a4)
{
  int v5; // r12d
  unsigned int v7; // esi
  __int64 v8; // rdi
  __int64 v9; // r8
  unsigned int v10; // ecx
  __int64 *v11; // rax
  __int64 v12; // rdx
  __int64 result; // rax
  int v14; // r11d
  __int64 *v15; // r10
  int v16; // ecx
  int v17; // ecx
  int v18; // eax
  int v19; // esi
  __int64 v20; // r8
  unsigned int v21; // edx
  __int64 v22; // rdi
  int v23; // r10d
  __int64 *v24; // r9
  int v25; // eax
  int v26; // edx
  __int64 v27; // rdi
  int v28; // r9d
  unsigned int v29; // r14d
  __int64 *v30; // r8
  __int64 v31; // rsi

  v5 = a3;
  v7 = *(_DWORD *)(a1 + 104);
  if ( a4 )
    v5 = a3 | 2;
  v8 = a1 + 80;
  if ( !v7 )
  {
    ++*(_QWORD *)(a1 + 80);
    goto LABEL_18;
  }
  v9 = *(_QWORD *)(a1 + 88);
  v10 = (v7 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v11 = (__int64 *)(v9 + 16LL * v10);
  v12 = *v11;
  if ( *v11 == a2 )
    goto LABEL_5;
  v14 = 1;
  v15 = 0;
  while ( v12 != -8 )
  {
    if ( !v15 && v12 == -16 )
      v15 = v11;
    v10 = (v7 - 1) & (v14 + v10);
    v11 = (__int64 *)(v9 + 16LL * v10);
    v12 = *v11;
    if ( *v11 == a2 )
      goto LABEL_5;
    ++v14;
  }
  v16 = *(_DWORD *)(a1 + 96);
  if ( v15 )
    v11 = v15;
  ++*(_QWORD *)(a1 + 80);
  v17 = v16 + 1;
  if ( 4 * v17 >= 3 * v7 )
  {
LABEL_18:
    sub_12DDEF0(v8, 2 * v7);
    v18 = *(_DWORD *)(a1 + 104);
    if ( v18 )
    {
      v19 = v18 - 1;
      v20 = *(_QWORD *)(a1 + 88);
      v21 = (v18 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v17 = *(_DWORD *)(a1 + 96) + 1;
      v11 = (__int64 *)(v20 + 16LL * v21);
      v22 = *v11;
      if ( *v11 != a2 )
      {
        v23 = 1;
        v24 = 0;
        while ( v22 != -8 )
        {
          if ( !v24 && v22 == -16 )
            v24 = v11;
          v21 = v19 & (v23 + v21);
          v11 = (__int64 *)(v20 + 16LL * v21);
          v22 = *v11;
          if ( *v11 == a2 )
            goto LABEL_14;
          ++v23;
        }
        if ( v24 )
          v11 = v24;
      }
      goto LABEL_14;
    }
    goto LABEL_46;
  }
  if ( v7 - *(_DWORD *)(a1 + 100) - v17 <= v7 >> 3 )
  {
    sub_12DDEF0(v8, v7);
    v25 = *(_DWORD *)(a1 + 104);
    if ( v25 )
    {
      v26 = v25 - 1;
      v27 = *(_QWORD *)(a1 + 88);
      v28 = 1;
      v29 = (v25 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v30 = 0;
      v17 = *(_DWORD *)(a1 + 96) + 1;
      v11 = (__int64 *)(v27 + 16LL * v29);
      v31 = *v11;
      if ( *v11 != a2 )
      {
        while ( v31 != -8 )
        {
          if ( v31 == -16 && !v30 )
            v30 = v11;
          v29 = v26 & (v28 + v29);
          v11 = (__int64 *)(v27 + 16LL * v29);
          v31 = *v11;
          if ( *v11 == a2 )
            goto LABEL_14;
          ++v28;
        }
        if ( v30 )
          v11 = v30;
      }
      goto LABEL_14;
    }
LABEL_46:
    ++*(_DWORD *)(a1 + 96);
    BUG();
  }
LABEL_14:
  *(_DWORD *)(a1 + 96) = v17;
  if ( *v11 != -8 )
    --*(_DWORD *)(a1 + 100);
  *v11 = a2;
  *((_DWORD *)v11 + 2) = 0;
LABEL_5:
  *((_DWORD *)v11 + 2) = v5;
  result = *(unsigned int *)(a1 + 8);
  if ( (unsigned int)result >= *(_DWORD *)(a1 + 12) )
  {
    sub_16CD150(a1, a1 + 16, 0, 8);
    result = *(unsigned int *)(a1 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a1 + 8 * result) = a2;
  ++*(_DWORD *)(a1 + 8);
  return result;
}
