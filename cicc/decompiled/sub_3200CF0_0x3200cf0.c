// Function: sub_3200CF0
// Address: 0x3200cf0
//
__int64 __fastcall sub_3200CF0(__int64 a1, __int64 a2)
{
  _QWORD *v3; // rax
  __int64 v4; // rax
  __int64 v5; // r13
  __int64 v6; // rax
  unsigned __int64 v7; // rbx
  unsigned int v8; // esi
  int v9; // r11d
  __int64 v10; // rdi
  unsigned __int64 *v11; // r10
  unsigned int v12; // ecx
  __int64 *v13; // rdx
  __int64 result; // rax
  int v15; // eax
  int v16; // edx
  int v17; // eax
  int v18; // ecx
  __int64 v19; // rdi
  unsigned int v20; // eax
  unsigned __int64 v21; // rsi
  int v22; // r9d
  unsigned __int64 *v23; // r8
  int v24; // eax
  int v25; // eax
  __int64 v26; // rsi
  int v27; // r8d
  unsigned __int64 *v28; // rdi
  unsigned int v29; // r13d
  unsigned __int64 v30; // rcx

  if ( !a2 )
    goto LABEL_5;
  v3 = *(_QWORD **)a2;
  if ( !*(_QWORD *)a2 )
  {
    if ( (*(_BYTE *)(a2 + 9) & 0x70) != 0x20 || *(char *)(a2 + 8) < 0 )
      BUG();
    *(_BYTE *)(a2 + 8) |= 8u;
    v3 = sub_E807D0(*(_QWORD *)(a2 + 24));
    *(_QWORD *)a2 = v3;
  }
  v4 = v3[1];
  if ( *(_DWORD *)(v4 + 144) )
LABEL_5:
    v5 = 0;
  else
    v5 = *(_QWORD *)(v4 + 160);
  v6 = sub_31DA6B0(*(_QWORD *)(a1 + 8));
  v7 = sub_E6E2A0(*(_QWORD **)(*(_QWORD *)(a1 + 528) + 8LL), *(_QWORD *)(v6 + 392), v5, 0xFFFFFFFF);
  (*(void (__fastcall **)(_QWORD, unsigned __int64, _QWORD))(**(_QWORD **)(a1 + 528) + 176LL))(
    *(_QWORD *)(a1 + 528),
    v7,
    0);
  v8 = *(_DWORD *)(a1 + 1040);
  if ( !v8 )
  {
    ++*(_QWORD *)(a1 + 1016);
    goto LABEL_25;
  }
  v9 = 1;
  v10 = *(_QWORD *)(a1 + 1024);
  v11 = 0;
  v12 = (v8 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
  v13 = (__int64 *)(v10 + 8LL * v12);
  result = *v13;
  if ( v7 == *v13 )
    return result;
  while ( result != -4096 )
  {
    if ( v11 || result != -8192 )
      v13 = (__int64 *)v11;
    v12 = (v8 - 1) & (v9 + v12);
    result = *(_QWORD *)(v10 + 8LL * v12);
    if ( v7 == result )
      return result;
    ++v9;
    v11 = (unsigned __int64 *)v13;
    v13 = (__int64 *)(v10 + 8LL * v12);
  }
  v15 = *(_DWORD *)(a1 + 1032);
  if ( !v11 )
    v11 = (unsigned __int64 *)v13;
  ++*(_QWORD *)(a1 + 1016);
  v16 = v15 + 1;
  if ( 4 * (v15 + 1) >= 3 * v8 )
  {
LABEL_25:
    sub_3200B20(a1 + 1016, 2 * v8);
    v17 = *(_DWORD *)(a1 + 1040);
    if ( v17 )
    {
      v18 = v17 - 1;
      v19 = *(_QWORD *)(a1 + 1024);
      v20 = (v17 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
      v11 = (unsigned __int64 *)(v19 + 8LL * v20);
      v21 = *v11;
      v16 = *(_DWORD *)(a1 + 1032) + 1;
      if ( v7 != *v11 )
      {
        v22 = 1;
        v23 = 0;
        while ( v21 != -4096 )
        {
          if ( v21 == -8192 && !v23 )
            v23 = v11;
          v20 = v18 & (v22 + v20);
          v11 = (unsigned __int64 *)(v19 + 8LL * v20);
          v21 = *v11;
          if ( v7 == *v11 )
            goto LABEL_21;
          ++v22;
        }
        if ( v23 )
          v11 = v23;
      }
      goto LABEL_21;
    }
    goto LABEL_49;
  }
  if ( v8 - *(_DWORD *)(a1 + 1036) - v16 <= v8 >> 3 )
  {
    sub_3200B20(a1 + 1016, v8);
    v24 = *(_DWORD *)(a1 + 1040);
    if ( v24 )
    {
      v25 = v24 - 1;
      v26 = *(_QWORD *)(a1 + 1024);
      v27 = 1;
      v28 = 0;
      v29 = v25 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
      v11 = (unsigned __int64 *)(v26 + 8LL * v29);
      v30 = *v11;
      v16 = *(_DWORD *)(a1 + 1032) + 1;
      if ( v7 != *v11 )
      {
        while ( v30 != -4096 )
        {
          if ( v30 == -8192 && !v28 )
            v28 = v11;
          v29 = v25 & (v27 + v29);
          v11 = (unsigned __int64 *)(v26 + 8LL * v29);
          v30 = *v11;
          if ( v7 == *v11 )
            goto LABEL_21;
          ++v27;
        }
        if ( v28 )
          v11 = v28;
      }
      goto LABEL_21;
    }
LABEL_49:
    ++*(_DWORD *)(a1 + 1032);
    BUG();
  }
LABEL_21:
  *(_DWORD *)(a1 + 1032) = v16;
  if ( *v11 != -4096 )
    --*(_DWORD *)(a1 + 1036);
  *v11 = v7;
  return sub_31F7AE0(a1);
}
