// Function: sub_15458C0
// Address: 0x15458c0
//
__int64 __fastcall sub_15458C0(__int64 a1, int a2, __int64 a3)
{
  __int64 v5; // rdi
  unsigned int v7; // esi
  __int64 v8; // rcx
  unsigned int v9; // r15d
  unsigned int v10; // edx
  __int64 result; // rax
  __int64 v12; // r10
  int v13; // r11d
  __int64 v14; // r14
  int v15; // eax
  int v16; // edx
  int v17; // eax
  int v18; // ecx
  __int64 v19; // rdi
  unsigned int v20; // eax
  __int64 v21; // rsi
  int v22; // r9d
  __int64 v23; // r8
  int v24; // eax
  int v25; // eax
  __int64 v26; // rsi
  int v27; // r8d
  unsigned int v28; // r15d
  __int64 v29; // rdi
  __int64 v30; // rcx
  __int64 v31[7]; // [rsp+8h] [rbp-38h] BYREF

  v5 = a1 + 256;
  v7 = *(_DWORD *)(a1 + 280);
  if ( !v7 )
  {
    ++*(_QWORD *)(a1 + 256);
    goto LABEL_16;
  }
  v8 = *(_QWORD *)(a1 + 264);
  v9 = ((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4);
  v10 = (v7 - 1) & v9;
  result = v8 + 16LL * v10;
  v12 = *(_QWORD *)result;
  if ( a3 != *(_QWORD *)result )
  {
    v13 = 1;
    v14 = 0;
    while ( v12 != -4 )
    {
      if ( !v14 && v12 == -8 )
        v14 = result;
      v10 = (v7 - 1) & (v13 + v10);
      result = v8 + 16LL * v10;
      v12 = *(_QWORD *)result;
      if ( a3 == *(_QWORD *)result )
        goto LABEL_3;
      ++v13;
    }
    if ( !v14 )
      v14 = result;
    v15 = *(_DWORD *)(a1 + 272);
    ++*(_QWORD *)(a1 + 256);
    v16 = v15 + 1;
    if ( 4 * (v15 + 1) < 3 * v7 )
    {
      if ( v7 - *(_DWORD *)(a1 + 276) - v16 > v7 >> 3 )
      {
LABEL_11:
        *(_DWORD *)(a1 + 272) = v16;
        if ( *(_QWORD *)v14 != -4 )
          --*(_DWORD *)(a1 + 276);
        *(_QWORD *)v14 = a3;
        *(_QWORD *)(v14 + 8) = 0;
        goto LABEL_14;
      }
      sub_1542590(v5, v7);
      v24 = *(_DWORD *)(a1 + 280);
      if ( v24 )
      {
        v25 = v24 - 1;
        v26 = *(_QWORD *)(a1 + 264);
        v27 = 1;
        v28 = v25 & v9;
        v16 = *(_DWORD *)(a1 + 272) + 1;
        v29 = 0;
        v14 = v26 + 16LL * v28;
        v30 = *(_QWORD *)v14;
        if ( a3 != *(_QWORD *)v14 )
        {
          while ( v30 != -4 )
          {
            if ( !v29 && v30 == -8 )
              v29 = v14;
            v28 = v25 & (v27 + v28);
            v14 = v26 + 16LL * v28;
            v30 = *(_QWORD *)v14;
            if ( a3 == *(_QWORD *)v14 )
              goto LABEL_11;
            ++v27;
          }
          if ( v29 )
            v14 = v29;
        }
        goto LABEL_11;
      }
LABEL_45:
      ++*(_DWORD *)(a1 + 272);
      BUG();
    }
LABEL_16:
    sub_1542590(v5, 2 * v7);
    v17 = *(_DWORD *)(a1 + 280);
    if ( v17 )
    {
      v18 = v17 - 1;
      v19 = *(_QWORD *)(a1 + 264);
      v20 = (v17 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
      v16 = *(_DWORD *)(a1 + 272) + 1;
      v14 = v19 + 16LL * v20;
      v21 = *(_QWORD *)v14;
      if ( a3 != *(_QWORD *)v14 )
      {
        v22 = 1;
        v23 = 0;
        while ( v21 != -4 )
        {
          if ( !v23 && v21 == -8 )
            v23 = v14;
          v20 = v18 & (v22 + v20);
          v14 = v19 + 16LL * v20;
          v21 = *(_QWORD *)v14;
          if ( a3 == *(_QWORD *)v14 )
            goto LABEL_11;
          ++v22;
        }
        if ( v23 )
          v14 = v23;
      }
      goto LABEL_11;
    }
    goto LABEL_45;
  }
LABEL_3:
  if ( !*(_DWORD *)(result + 12) )
  {
    v14 = result;
LABEL_14:
    v31[0] = a3;
    sub_153F6C0(a1 + 208, v31);
    *(_DWORD *)(v14 + 8) = a2;
    *(_DWORD *)(v14 + 12) = (__int64)(*(_QWORD *)(a1 + 216) - *(_QWORD *)(a1 + 208)) >> 3;
    return sub_15445A0(a1, *(_QWORD *)(a3 + 136));
  }
  return result;
}
