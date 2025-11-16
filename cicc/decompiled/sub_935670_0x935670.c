// Function: sub_935670
// Address: 0x935670
//
__int64 __fastcall sub_935670(__int64 a1, __int64 a2)
{
  __int64 v4; // rbx
  __int64 v5; // r15
  unsigned int v6; // esi
  int v7; // r11d
  __int64 v8; // r8
  _QWORD *v9; // rdx
  unsigned int v10; // edi
  _QWORD *v11; // rax
  __int64 v12; // rcx
  _QWORD *v13; // rax
  __int64 v14; // rdi
  __int64 v15; // rcx
  int v16; // esi
  _QWORD *v17; // rsi
  int v19; // eax
  int v20; // ecx
  int v21; // eax
  int v22; // esi
  __int64 v23; // rdi
  unsigned int v24; // eax
  __int64 v25; // r8
  int v26; // r10d
  _QWORD *v27; // r9
  int v28; // eax
  int v29; // eax
  int v30; // r9d
  _QWORD *v31; // r8
  __int64 v32; // rdi
  unsigned int v33; // r14d
  __int64 v34; // rsi

  v4 = *(_QWORD *)(a2 + 72);
  v5 = *(_QWORD *)(a2 + 80);
  if ( *(_BYTE *)(v4 + 40) != 16 )
    sub_91B8A0("associated switch for case statement not found!", (_DWORD *)a2, 1);
  v6 = *(_DWORD *)(a1 + 520);
  if ( !v6 )
  {
    ++*(_QWORD *)(a1 + 496);
    goto LABEL_30;
  }
  v7 = 1;
  v8 = *(_QWORD *)(a1 + 504);
  v9 = 0;
  v10 = (v6 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
  v11 = (_QWORD *)(v8 + 32LL * v10);
  v12 = *v11;
  if ( v4 == *v11 )
  {
LABEL_5:
    v13 = v11 + 1;
    goto LABEL_6;
  }
  while ( v12 != -4096 )
  {
    if ( !v9 && v12 == -8192 )
      v9 = v11;
    v10 = (v6 - 1) & (v7 + v10);
    v11 = (_QWORD *)(v8 + 32LL * v10);
    v12 = *v11;
    if ( v4 == *v11 )
      goto LABEL_5;
    ++v7;
  }
  if ( !v9 )
    v9 = v11;
  v19 = *(_DWORD *)(a1 + 512);
  ++*(_QWORD *)(a1 + 496);
  v20 = v19 + 1;
  if ( 4 * (v19 + 1) >= 3 * v6 )
  {
LABEL_30:
    sub_935440(a1 + 496, 2 * v6);
    v21 = *(_DWORD *)(a1 + 520);
    if ( v21 )
    {
      v22 = v21 - 1;
      v23 = *(_QWORD *)(a1 + 504);
      v24 = (v21 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v20 = *(_DWORD *)(a1 + 512) + 1;
      v9 = (_QWORD *)(v23 + 32LL * v24);
      v25 = *v9;
      if ( v4 != *v9 )
      {
        v26 = 1;
        v27 = 0;
        while ( v25 != -4096 )
        {
          if ( !v27 && v25 == -8192 )
            v27 = v9;
          v24 = v22 & (v26 + v24);
          v9 = (_QWORD *)(v23 + 32LL * v24);
          v25 = *v9;
          if ( v4 == *v9 )
            goto LABEL_25;
          ++v26;
        }
        if ( v27 )
          v9 = v27;
      }
      goto LABEL_25;
    }
    goto LABEL_53;
  }
  if ( v6 - *(_DWORD *)(a1 + 516) - v20 <= v6 >> 3 )
  {
    sub_935440(a1 + 496, v6);
    v28 = *(_DWORD *)(a1 + 520);
    if ( v28 )
    {
      v29 = v28 - 1;
      v30 = 1;
      v31 = 0;
      v32 = *(_QWORD *)(a1 + 504);
      v33 = v29 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v20 = *(_DWORD *)(a1 + 512) + 1;
      v9 = (_QWORD *)(v32 + 32LL * v33);
      v34 = *v9;
      if ( v4 != *v9 )
      {
        while ( v34 != -4096 )
        {
          if ( v34 == -8192 && !v31 )
            v31 = v9;
          v33 = v29 & (v30 + v33);
          v9 = (_QWORD *)(v32 + 32LL * v33);
          v34 = *v9;
          if ( v4 == *v9 )
            goto LABEL_25;
          ++v30;
        }
        if ( v31 )
          v9 = v31;
      }
      goto LABEL_25;
    }
LABEL_53:
    ++*(_DWORD *)(a1 + 512);
    BUG();
  }
LABEL_25:
  *(_DWORD *)(a1 + 512) = v20;
  if ( *v9 != -4096 )
    --*(_DWORD *)(a1 + 516);
  *v9 = v4;
  v13 = v9 + 1;
  v9[1] = 0;
  v9[2] = 0;
  v9[3] = 0;
LABEL_6:
  v14 = *(_QWORD *)(v5 + 8);
  if ( v14 )
  {
    v15 = *(_QWORD *)(*(_QWORD *)(v4 + 80) + 16LL);
    if ( !v15 )
      goto LABEL_28;
    v16 = 0;
    while ( v14 != *(_QWORD *)(v15 + 8) )
    {
      v15 = *(_QWORD *)(v15 + 32);
      ++v16;
      if ( !v15 )
        goto LABEL_28;
    }
    v17 = *(_QWORD **)(*v13 + 8LL * v16);
  }
  else
  {
    v17 = *(_QWORD **)(v13[1] - 8LL);
  }
  if ( !v17 )
LABEL_28:
    sub_91B8A0("basic block for case statement not found!", (_DWORD *)a2, 1);
  return sub_92FEA0(a1, v17, 0);
}
