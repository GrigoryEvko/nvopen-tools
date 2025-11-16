// Function: sub_31BAED0
// Address: 0x31baed0
//
__int64 __fastcall sub_31BAED0(__int64 a1, __int64 a2)
{
  unsigned int v3; // esi
  __int64 v5; // rdi
  int v6; // r9d
  __int64 *v7; // r14
  unsigned int v8; // ecx
  __int64 *v9; // rax
  __int64 v10; // rdx
  __int64 result; // rax
  int v12; // eax
  int v13; // edx
  __int64 v14; // rdi
  int v15; // eax
  int v16; // ecx
  __int64 v17; // rdi
  unsigned int v18; // eax
  __int64 v19; // rsi
  int v20; // r9d
  __int64 *v21; // r8
  int v22; // eax
  int v23; // eax
  __int64 v24; // rsi
  int v25; // r8d
  unsigned int v26; // r13d
  __int64 *v27; // rdi
  __int64 v28; // rcx

  v3 = *(_DWORD *)(a1 + 24);
  if ( !v3 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_25;
  }
  v5 = *(_QWORD *)(a1 + 8);
  v6 = 1;
  v7 = 0;
  v8 = (v3 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v9 = (__int64 *)(v5 + 16LL * v8);
  v10 = *v9;
  if ( *v9 == a2 )
    return v9[1];
  while ( v10 != -4096 )
  {
    if ( !v7 && v10 == -8192 )
      v7 = v9;
    v8 = (v3 - 1) & (v6 + v8);
    v9 = (__int64 *)(v5 + 16LL * v8);
    v10 = *v9;
    if ( *v9 == a2 )
      return v9[1];
    ++v6;
  }
  if ( !v7 )
    v7 = v9;
  v12 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v13 = v12 + 1;
  if ( 4 * (v12 + 1) >= 3 * v3 )
  {
LABEL_25:
    sub_31BACD0(a1, 2 * v3);
    v15 = *(_DWORD *)(a1 + 24);
    if ( v15 )
    {
      v16 = v15 - 1;
      v17 = *(_QWORD *)(a1 + 8);
      v18 = (v15 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v13 = *(_DWORD *)(a1 + 16) + 1;
      v7 = (__int64 *)(v17 + 16LL * v18);
      v19 = *v7;
      if ( *v7 != a2 )
      {
        v20 = 1;
        v21 = 0;
        while ( v19 != -4096 )
        {
          if ( !v21 && v19 == -8192 )
            v21 = v7;
          v18 = v16 & (v20 + v18);
          v7 = (__int64 *)(v17 + 16LL * v18);
          v19 = *v7;
          if ( *v7 == a2 )
            goto LABEL_15;
          ++v20;
        }
        if ( v21 )
          v7 = v21;
      }
      goto LABEL_15;
    }
    goto LABEL_48;
  }
  if ( v3 - *(_DWORD *)(a1 + 20) - v13 <= v3 >> 3 )
  {
    sub_31BACD0(a1, v3);
    v22 = *(_DWORD *)(a1 + 24);
    if ( v22 )
    {
      v23 = v22 - 1;
      v24 = *(_QWORD *)(a1 + 8);
      v25 = 1;
      v26 = v23 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v27 = 0;
      v13 = *(_DWORD *)(a1 + 16) + 1;
      v7 = (__int64 *)(v24 + 16LL * v26);
      v28 = *v7;
      if ( *v7 != a2 )
      {
        while ( v28 != -4096 )
        {
          if ( !v27 && v28 == -8192 )
            v27 = v7;
          v26 = v23 & (v25 + v26);
          v7 = (__int64 *)(v24 + 16LL * v26);
          v28 = *v7;
          if ( *v7 == a2 )
            goto LABEL_15;
          ++v25;
        }
        if ( v27 )
          v7 = v27;
      }
      goto LABEL_15;
    }
LABEL_48:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
LABEL_15:
  *(_DWORD *)(a1 + 16) = v13;
  if ( *v7 != -4096 )
    --*(_DWORD *)(a1 + 20);
  *v7 = a2;
  v7[1] = 0;
  if ( (unsigned __int8)sub_31B84B0(a2) )
  {
    result = sub_22077B0(0x78u);
    if ( result )
    {
      *(_QWORD *)(result + 8) = a2;
      *(_QWORD *)(result + 16) = 1;
      *(_BYTE *)(result + 24) = 0;
      *(_QWORD *)result = &unk_4A349D0;
      *(_QWORD *)(result + 32) = 0;
      *(_QWORD *)(result + 40) = 0;
      *(_QWORD *)(result + 48) = 0;
      *(_QWORD *)(result + 56) = 0;
      *(_QWORD *)(result + 64) = 0;
      *(_QWORD *)(result + 72) = 0;
      *(_DWORD *)(result + 80) = 0;
      *(_QWORD *)(result + 88) = 0;
      *(_QWORD *)(result + 96) = 0;
      *(_QWORD *)(result + 104) = 0;
      *(_DWORD *)(result + 112) = 0;
    }
  }
  else
  {
    result = sub_22077B0(0x28u);
    if ( result )
    {
      *(_QWORD *)(result + 8) = a2;
      *(_QWORD *)(result + 16) = 0;
      *(_BYTE *)(result + 24) = 0;
      *(_QWORD *)result = &unk_4A34A00;
      *(_QWORD *)(result + 32) = 0;
    }
  }
  v14 = v7[1];
  v7[1] = result;
  if ( v14 )
  {
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v14 + 8LL))(v14);
    return v7[1];
  }
  return result;
}
