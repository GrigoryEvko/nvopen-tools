// Function: sub_AE4AC0
// Address: 0xae4ac0
//
__int64 __fastcall sub_AE4AC0(__int64 a1, __int64 a2)
{
  __int64 v4; // rbx
  __int64 v5; // rcx
  __int64 v6; // rsi
  unsigned __int64 v7; // r8
  int v8; // r11d
  __int64 *v9; // rdx
  unsigned int v10; // edi
  __int64 *v11; // rax
  __int64 v12; // r9
  __int64 *v13; // rbx
  __int64 result; // rax
  int v15; // eax
  __int64 v16; // rax
  __int64 v17; // rax
  int v18; // eax
  unsigned int v19; // eax
  __int64 v20; // rdi
  int v21; // r10d
  int v22; // eax
  int v23; // eax
  __int64 v24; // rdi
  unsigned int v25; // r14d
  __int64 v26; // [rsp+8h] [rbp-28h]

  v4 = *(_QWORD *)(a1 + 488);
  if ( v4 )
  {
    v5 = *(_QWORD *)(v4 + 8);
    v6 = *(unsigned int *)(v4 + 24);
  }
  else
  {
    v17 = sub_22077B0(32);
    v4 = v17;
    if ( v17 )
    {
      *(_QWORD *)v17 = 0;
      *(_QWORD *)(v17 + 8) = 0;
      *(_QWORD *)(v17 + 16) = 0;
      *(_DWORD *)(v17 + 24) = 0;
      *(_QWORD *)(a1 + 488) = v17;
      goto LABEL_24;
    }
    v5 = MEMORY[8];
    v6 = MEMORY[0x18];
    *(_QWORD *)(a1 + 488) = 0;
  }
  if ( !(_DWORD)v6 )
  {
LABEL_24:
    ++*(_QWORD *)v4;
    LODWORD(v6) = 0;
    goto LABEL_25;
  }
  v7 = (unsigned int)(v6 - 1);
  v8 = 1;
  v9 = 0;
  v10 = v7 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v11 = (__int64 *)(v5 + 16LL * v10);
  v12 = *v11;
  if ( a2 == *v11 )
  {
LABEL_5:
    v13 = v11 + 1;
    result = v11[1];
    if ( result )
      return result;
    goto LABEL_20;
  }
  while ( v12 != -4096 )
  {
    if ( !v9 && v12 == -8192 )
      v9 = v11;
    v10 = v7 & (v8 + v10);
    v11 = (__int64 *)(v5 + 16LL * v10);
    v12 = *v11;
    if ( a2 == *v11 )
      goto LABEL_5;
    ++v8;
  }
  if ( !v9 )
    v9 = v11;
  v15 = *(_DWORD *)(v4 + 16);
  ++*(_QWORD *)v4;
  v5 = (unsigned int)(v15 + 1);
  if ( 4 * (int)v5 >= (unsigned int)(3 * v6) )
  {
LABEL_25:
    sub_AE48E0(v4, 2 * v6);
    v18 = *(_DWORD *)(v4 + 24);
    if ( v18 )
    {
      v6 = (unsigned int)(v18 - 1);
      v7 = *(_QWORD *)(v4 + 8);
      v19 = v6 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v5 = (unsigned int)(*(_DWORD *)(v4 + 16) + 1);
      v9 = (__int64 *)(v7 + 16LL * v19);
      v20 = *v9;
      if ( a2 != *v9 )
      {
        v21 = 1;
        v12 = 0;
        while ( v20 != -4096 )
        {
          if ( !v12 && v20 == -8192 )
            v12 = (__int64)v9;
          v19 = v6 & (v21 + v19);
          v9 = (__int64 *)(v7 + 16LL * v19);
          v20 = *v9;
          if ( a2 == *v9 )
            goto LABEL_17;
          ++v21;
        }
        if ( v12 )
          v9 = (__int64 *)v12;
      }
      goto LABEL_17;
    }
    goto LABEL_50;
  }
  if ( (int)v6 - *(_DWORD *)(v4 + 20) - (int)v5 <= (unsigned int)v6 >> 3 )
  {
    sub_AE48E0(v4, v6);
    v22 = *(_DWORD *)(v4 + 24);
    if ( v22 )
    {
      v23 = v22 - 1;
      v24 = *(_QWORD *)(v4 + 8);
      v12 = 1;
      v25 = v23 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v7 = 0;
      v5 = (unsigned int)(*(_DWORD *)(v4 + 16) + 1);
      v9 = (__int64 *)(v24 + 16LL * v25);
      v6 = *v9;
      if ( a2 != *v9 )
      {
        while ( v6 != -4096 )
        {
          if ( v6 == -8192 && !v7 )
            v7 = (unsigned __int64)v9;
          v25 = v23 & (v12 + v25);
          v9 = (__int64 *)(v24 + 16LL * v25);
          v6 = *v9;
          if ( a2 == *v9 )
            goto LABEL_17;
          v12 = (unsigned int)(v12 + 1);
        }
        if ( v7 )
          v9 = (__int64 *)v7;
      }
      goto LABEL_17;
    }
LABEL_50:
    ++*(_DWORD *)(v4 + 16);
    BUG();
  }
LABEL_17:
  *(_DWORD *)(v4 + 16) = v5;
  if ( *v9 != -4096 )
    --*(_DWORD *)(v4 + 20);
  *v9 = a2;
  v13 = v9 + 1;
  v9[1] = 0;
LABEL_20:
  v16 = malloc(16LL * *(unsigned int *)(a2 + 12) + 24, v6, v9, v5, v7, v12);
  if ( !v16 )
    sub_C64F00("Allocation failed");
  *v13 = v16;
  v26 = v16;
  sub_AE5030(v16, a2, a1);
  return v26;
}
