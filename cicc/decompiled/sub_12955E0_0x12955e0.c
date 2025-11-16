// Function: sub_12955E0
// Address: 0x12955e0
//
__int64 __fastcall sub_12955E0(__int64 a1, __int64 a2)
{
  __int64 v4; // rbx
  __int64 v5; // r15
  unsigned int v6; // esi
  __int64 v7; // rdi
  unsigned int v8; // ecx
  _QWORD *v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rsi
  __int64 v12; // rdx
  int v13; // ecx
  _QWORD *v14; // rsi
  int v16; // r11d
  _QWORD *v17; // r10
  int v18; // edi
  int v19; // ecx
  int v20; // eax
  int v21; // esi
  __int64 v22; // rdi
  unsigned int v23; // edx
  __int64 v24; // r8
  int v25; // r10d
  _QWORD *v26; // r9
  int v27; // eax
  int v28; // edx
  int v29; // r9d
  _QWORD *v30; // r8
  __int64 v31; // rdi
  unsigned int v32; // r14d
  __int64 v33; // rsi

  v4 = *(_QWORD *)(a2 + 72);
  v5 = *(_QWORD *)(a2 + 80);
  if ( *(_BYTE *)(v4 + 40) != 16 )
    sub_127B550("associated switch for case statement not found!", (_DWORD *)a2, 1);
  v6 = *(_DWORD *)(a1 + 432);
  if ( !v6 )
  {
    ++*(_QWORD *)(a1 + 408);
    goto LABEL_25;
  }
  v7 = *(_QWORD *)(a1 + 416);
  v8 = (v6 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
  v9 = (_QWORD *)(v7 + 32LL * v8);
  v10 = *v9;
  if ( v4 == *v9 )
    goto LABEL_5;
  v16 = 1;
  v17 = 0;
  while ( v10 != -8 )
  {
    if ( !v17 && v10 == -16 )
      v17 = v9;
    v8 = (v6 - 1) & (v16 + v8);
    v9 = (_QWORD *)(v7 + 32LL * v8);
    v10 = *v9;
    if ( v4 == *v9 )
      goto LABEL_5;
    ++v16;
  }
  v18 = *(_DWORD *)(a1 + 424);
  if ( v17 )
    v9 = v17;
  ++*(_QWORD *)(a1 + 408);
  v19 = v18 + 1;
  if ( 4 * (v18 + 1) >= 3 * v6 )
  {
LABEL_25:
    sub_12953C0(a1 + 408, 2 * v6);
    v20 = *(_DWORD *)(a1 + 432);
    if ( v20 )
    {
      v21 = v20 - 1;
      v22 = *(_QWORD *)(a1 + 416);
      v23 = (v20 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v19 = *(_DWORD *)(a1 + 424) + 1;
      v9 = (_QWORD *)(v22 + 32LL * v23);
      v24 = *v9;
      if ( v4 != *v9 )
      {
        v25 = 1;
        v26 = 0;
        while ( v24 != -8 )
        {
          if ( !v26 && v24 == -16 )
            v26 = v9;
          v23 = v21 & (v25 + v23);
          v9 = (_QWORD *)(v22 + 32LL * v23);
          v24 = *v9;
          if ( v4 == *v9 )
            goto LABEL_21;
          ++v25;
        }
        if ( v26 )
          v9 = v26;
      }
      goto LABEL_21;
    }
    goto LABEL_53;
  }
  if ( v6 - *(_DWORD *)(a1 + 428) - v19 <= v6 >> 3 )
  {
    sub_12953C0(a1 + 408, v6);
    v27 = *(_DWORD *)(a1 + 432);
    if ( v27 )
    {
      v28 = v27 - 1;
      v29 = 1;
      v30 = 0;
      v31 = *(_QWORD *)(a1 + 416);
      v32 = (v27 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v19 = *(_DWORD *)(a1 + 424) + 1;
      v9 = (_QWORD *)(v31 + 32LL * v32);
      v33 = *v9;
      if ( v4 != *v9 )
      {
        while ( v33 != -8 )
        {
          if ( v33 == -16 && !v30 )
            v30 = v9;
          v32 = v28 & (v29 + v32);
          v9 = (_QWORD *)(v31 + 32LL * v32);
          v33 = *v9;
          if ( v4 == *v9 )
            goto LABEL_21;
          ++v29;
        }
        if ( v30 )
          v9 = v30;
      }
      goto LABEL_21;
    }
LABEL_53:
    ++*(_DWORD *)(a1 + 424);
    BUG();
  }
LABEL_21:
  *(_DWORD *)(a1 + 424) = v19;
  if ( *v9 != -8 )
    --*(_DWORD *)(a1 + 428);
  *v9 = v4;
  v9[1] = 0;
  v9[2] = 0;
  v9[3] = 0;
LABEL_5:
  v11 = *(_QWORD *)(v5 + 8);
  if ( v11 )
  {
    v12 = *(_QWORD *)(*(_QWORD *)(v4 + 80) + 16LL);
    if ( !v12 )
      goto LABEL_14;
    v13 = 0;
    while ( v11 != *(_QWORD *)(v12 + 8) )
    {
      v12 = *(_QWORD *)(v12 + 32);
      ++v13;
      if ( !v12 )
        goto LABEL_14;
    }
    v14 = *(_QWORD **)(v9[1] + 8LL * v13);
  }
  else
  {
    v14 = *(_QWORD **)(v9[2] - 8LL);
  }
  if ( !v14 )
LABEL_14:
    sub_127B550("basic block for case statement not found!", (_DWORD *)a2, 1);
  return sub_1290AF0((_QWORD *)a1, v14, 0);
}
