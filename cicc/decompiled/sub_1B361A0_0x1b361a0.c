// Function: sub_1B361A0
// Address: 0x1b361a0
//
__int64 __fastcall sub_1B361A0(__int64 a1, __int64 a2)
{
  __int64 v4; // rdi
  unsigned int v5; // esi
  __int64 v6; // rcx
  unsigned int v7; // edx
  __int64 *v8; // rax
  __int64 v9; // r9
  int v10; // edx
  __int64 result; // rax
  int v12; // r11d
  __int64 *v13; // r13
  int v14; // eax
  int v15; // edx
  __int64 v16; // rbx
  int v17; // r12d
  int v18; // r12d
  int v19; // eax
  int v20; // ecx
  __int64 v21; // rdi
  unsigned int v22; // eax
  __int64 v23; // rsi
  int v24; // r9d
  __int64 *v25; // r8
  int v26; // eax
  int v27; // eax
  __int64 v28; // rsi
  __int64 *v29; // rdi
  unsigned int v30; // r12d
  int v31; // r8d
  __int64 v32; // rcx

  v4 = a1 + 928;
  v5 = *(_DWORD *)(a1 + 952);
  if ( !v5 )
  {
    ++*(_QWORD *)(a1 + 928);
    goto LABEL_25;
  }
  v6 = *(_QWORD *)(a1 + 936);
  v7 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v8 = (__int64 *)(v6 + 16LL * v7);
  v9 = *v8;
  if ( *v8 != a2 )
  {
    v12 = 1;
    v13 = 0;
    while ( v9 != -8 )
    {
      if ( !v13 && v9 == -16 )
        v13 = v8;
      v7 = (v5 - 1) & (v12 + v7);
      v8 = (__int64 *)(v6 + 16LL * v7);
      v9 = *v8;
      if ( *v8 == a2 )
        goto LABEL_3;
      ++v12;
    }
    if ( !v13 )
      v13 = v8;
    v14 = *(_DWORD *)(a1 + 944);
    ++*(_QWORD *)(a1 + 928);
    v15 = v14 + 1;
    if ( 4 * (v14 + 1) < 3 * v5 )
    {
      if ( v5 - *(_DWORD *)(a1 + 948) - v15 > v5 >> 3 )
      {
LABEL_11:
        *(_DWORD *)(a1 + 944) = v15;
        if ( *v13 != -8 )
          --*(_DWORD *)(a1 + 948);
        *v13 = a2;
        *((_DWORD *)v13 + 2) = 0;
        goto LABEL_14;
      }
      sub_137BFC0(v4, v5);
      v26 = *(_DWORD *)(a1 + 952);
      if ( v26 )
      {
        v27 = v26 - 1;
        v28 = *(_QWORD *)(a1 + 936);
        v29 = 0;
        v30 = v27 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v31 = 1;
        v15 = *(_DWORD *)(a1 + 944) + 1;
        v13 = (__int64 *)(v28 + 16LL * v30);
        v32 = *v13;
        if ( *v13 != a2 )
        {
          while ( v32 != -8 )
          {
            if ( v32 == -16 && !v29 )
              v29 = v13;
            v30 = v27 & (v31 + v30);
            v13 = (__int64 *)(v28 + 16LL * v30);
            v32 = *v13;
            if ( *v13 == a2 )
              goto LABEL_11;
            ++v31;
          }
          if ( v29 )
            v13 = v29;
        }
        goto LABEL_11;
      }
LABEL_54:
      ++*(_DWORD *)(a1 + 944);
      BUG();
    }
LABEL_25:
    sub_137BFC0(v4, 2 * v5);
    v19 = *(_DWORD *)(a1 + 952);
    if ( v19 )
    {
      v20 = v19 - 1;
      v21 = *(_QWORD *)(a1 + 936);
      v22 = (v19 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v15 = *(_DWORD *)(a1 + 944) + 1;
      v13 = (__int64 *)(v21 + 16LL * v22);
      v23 = *v13;
      if ( *v13 != a2 )
      {
        v24 = 1;
        v25 = 0;
        while ( v23 != -8 )
        {
          if ( !v25 && v23 == -16 )
            v25 = v13;
          v22 = v20 & (v24 + v22);
          v13 = (__int64 *)(v21 + 16LL * v22);
          v23 = *v13;
          if ( *v13 == a2 )
            goto LABEL_11;
          ++v24;
        }
        if ( v25 )
          v13 = v25;
      }
      goto LABEL_11;
    }
    goto LABEL_54;
  }
LABEL_3:
  v10 = *((_DWORD *)v8 + 2);
  if ( v10 )
    return (unsigned int)(v10 - 1);
  v13 = v8;
LABEL_14:
  v16 = *(_QWORD *)(a2 + 8);
  if ( v16 )
  {
    while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v16) + 16) - 25) > 9u )
    {
      v16 = *(_QWORD *)(v16 + 8);
      if ( !v16 )
        goto LABEL_23;
    }
    v17 = 0;
    while ( 1 )
    {
      v16 = *(_QWORD *)(v16 + 8);
      if ( !v16 )
        break;
      while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v16) + 16) - 25) <= 9u )
      {
        v16 = *(_QWORD *)(v16 + 8);
        ++v17;
        if ( !v16 )
          goto LABEL_20;
      }
    }
LABEL_20:
    result = (unsigned int)(v17 + 1);
    v18 = v17 + 2;
  }
  else
  {
LABEL_23:
    result = 0;
    v18 = 1;
  }
  *((_DWORD *)v13 + 2) = v18;
  return result;
}
