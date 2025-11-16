// Function: sub_15A9930
// Address: 0x15a9930
//
__int64 __fastcall sub_15A9930(__int64 a1, __int64 a2)
{
  __int64 v4; // rbx
  __int64 v5; // rdx
  unsigned int v6; // esi
  unsigned int v7; // ecx
  _QWORD *v8; // rax
  __int64 v9; // r8
  __int64 v10; // r14
  __int64 v12; // rax
  int v13; // eax
  int v14; // ecx
  __int64 v15; // rdi
  unsigned int v16; // eax
  int v17; // edx
  _QWORD *v18; // r15
  __int64 v19; // rsi
  int v20; // r9d
  _QWORD *v21; // r8
  int v22; // r10d
  int v23; // eax
  unsigned __int64 v24; // rbx
  int v25; // eax
  int v26; // eax
  __int64 v27; // rsi
  int v28; // r8d
  unsigned int v29; // r14d
  _QWORD *v30; // rdi
  __int64 v31; // rcx
  __int64 v32; // rax

  v4 = *(_QWORD *)(a1 + 400);
  if ( v4 )
  {
    v5 = *(_QWORD *)(v4 + 8);
    v6 = *(_DWORD *)(v4 + 24);
  }
  else
  {
    v12 = sub_22077B0(32);
    v4 = v12;
    if ( v12 )
    {
      *(_QWORD *)v12 = 0;
      *(_QWORD *)(v12 + 8) = 0;
      *(_QWORD *)(v12 + 16) = 0;
      *(_DWORD *)(v12 + 24) = 0;
      *(_QWORD *)(a1 + 400) = v12;
      goto LABEL_9;
    }
    v5 = MEMORY[8];
    v6 = MEMORY[0x18];
    *(_QWORD *)(a1 + 400) = 0;
  }
  if ( !v6 )
  {
LABEL_9:
    ++*(_QWORD *)v4;
    v6 = 0;
    goto LABEL_10;
  }
  v7 = (v6 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v8 = (_QWORD *)(v5 + 16LL * v7);
  v9 = *v8;
  if ( a2 != *v8 )
  {
    v22 = 1;
    v18 = 0;
    while ( v9 != -8 )
    {
      if ( v9 == -16 && !v18 )
        v18 = v8;
      v7 = (v6 - 1) & (v22 + v7);
      v8 = (_QWORD *)(v5 + 16LL * v7);
      v9 = *v8;
      if ( a2 == *v8 )
        goto LABEL_5;
      ++v22;
    }
    if ( !v18 )
      v18 = v8;
    v23 = *(_DWORD *)(v4 + 16);
    ++*(_QWORD *)v4;
    v17 = v23 + 1;
    if ( 4 * (v23 + 1) < 3 * v6 )
    {
      if ( v6 - *(_DWORD *)(v4 + 20) - v17 > v6 >> 3 )
      {
LABEL_23:
        *(_DWORD *)(v4 + 16) = v17;
        if ( *v18 != -8 )
          --*(_DWORD *)(v4 + 20);
        *v18 = a2;
        v18[1] = 0;
        goto LABEL_26;
      }
      sub_15A9770(v4, v6);
      v25 = *(_DWORD *)(v4 + 24);
      if ( v25 )
      {
        v26 = v25 - 1;
        v27 = *(_QWORD *)(v4 + 8);
        v28 = 1;
        v29 = v26 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v30 = 0;
        v17 = *(_DWORD *)(v4 + 16) + 1;
        v18 = (_QWORD *)(v27 + 16LL * v29);
        v31 = *v18;
        if ( a2 != *v18 )
        {
          while ( v31 != -8 )
          {
            if ( !v30 && v31 == -16 )
              v30 = v18;
            v29 = v26 & (v28 + v29);
            v18 = (_QWORD *)(v27 + 16LL * v29);
            v31 = *v18;
            if ( a2 == *v18 )
              goto LABEL_23;
            ++v28;
          }
          if ( v30 )
            v18 = v30;
        }
        goto LABEL_23;
      }
LABEL_55:
      ++*(_DWORD *)(v4 + 16);
      BUG();
    }
LABEL_10:
    sub_15A9770(v4, 2 * v6);
    v13 = *(_DWORD *)(v4 + 24);
    if ( v13 )
    {
      v14 = v13 - 1;
      v15 = *(_QWORD *)(v4 + 8);
      v16 = (v13 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v17 = *(_DWORD *)(v4 + 16) + 1;
      v18 = (_QWORD *)(v15 + 16LL * v16);
      v19 = *v18;
      if ( a2 != *v18 )
      {
        v20 = 1;
        v21 = 0;
        while ( v19 != -8 )
        {
          if ( !v21 && v19 == -16 )
            v21 = v18;
          v16 = v14 & (v20 + v16);
          v18 = (_QWORD *)(v15 + 16LL * v16);
          v19 = *v18;
          if ( a2 == *v18 )
            goto LABEL_23;
          ++v20;
        }
        if ( v21 )
          v18 = v21;
      }
      goto LABEL_23;
    }
    goto LABEL_55;
  }
LABEL_5:
  v10 = v8[1];
  if ( v10 )
    return v10;
  v18 = v8;
LABEL_26:
  v24 = 8LL * *(int *)(a2 + 12) + 16;
  v10 = malloc(v24);
  if ( !v10 )
  {
    if ( v24 || (v32 = malloc(1u)) == 0 )
      sub_16BD1C0("Allocation failed");
    else
      v10 = v32;
  }
  v18[1] = v10;
  sub_15AA470(v10, a2, a1);
  return v10;
}
