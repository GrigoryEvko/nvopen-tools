// Function: sub_2CF7710
// Address: 0x2cf7710
//
__int64 __fastcall sub_2CF7710(__int64 a1, __int64 a2)
{
  unsigned int v4; // esi
  __int64 v5; // r8
  int v6; // r10d
  __int64 *v7; // rdx
  unsigned int v8; // r13d
  unsigned int v9; // edi
  __int64 *v10; // rax
  __int64 v11; // rcx
  unsigned int v12; // r13d
  int v14; // eax
  int v15; // ecx
  __int64 v16; // rax
  char v17; // cl
  __int64 v18; // rdi
  int v19; // eax
  int v20; // esi
  __int64 v21; // r8
  unsigned int v22; // eax
  __int64 v23; // rdi
  int v24; // r10d
  __int64 *v25; // r9
  int v26; // eax
  int v27; // eax
  __int64 v28; // rdi
  __int64 *v29; // r8
  unsigned int v30; // r13d
  int v31; // r9d
  __int64 v32; // rsi
  __int64 v33[5]; // [rsp+8h] [rbp-28h] BYREF

  v4 = *(_DWORD *)(a2 + 24);
  v33[0] = a1;
  if ( v4 )
  {
    v5 = *(_QWORD *)(a2 + 8);
    v6 = 1;
    v7 = 0;
    v8 = ((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4);
    v9 = (v4 - 1) & v8;
    v10 = (__int64 *)(v5 + 16LL * v9);
    v11 = *v10;
    if ( a1 == *v10 )
      return *((unsigned __int8 *)v10 + 8);
    while ( v11 != -4096 )
    {
      if ( v11 == -8192 && !v7 )
        v7 = v10;
      v9 = (v4 - 1) & (v6 + v9);
      v10 = (__int64 *)(v5 + 16LL * v9);
      v11 = *v10;
      if ( a1 == *v10 )
        return *((unsigned __int8 *)v10 + 8);
      ++v6;
    }
    if ( !v7 )
      v7 = v10;
    v14 = *(_DWORD *)(a2 + 16);
    ++*(_QWORD *)a2;
    v15 = v14 + 1;
    if ( 4 * (v14 + 1) < 3 * v4 )
    {
      if ( v4 - *(_DWORD *)(a2 + 20) - v15 > v4 >> 3 )
        goto LABEL_15;
      sub_2CF72E0(a2, v4);
      v26 = *(_DWORD *)(a2 + 24);
      if ( v26 )
      {
        v27 = v26 - 1;
        v28 = *(_QWORD *)(a2 + 8);
        v29 = 0;
        v30 = v27 & v8;
        v31 = 1;
        v15 = *(_DWORD *)(a2 + 16) + 1;
        v7 = (__int64 *)(v28 + 16LL * v30);
        v32 = *v7;
        if ( a1 != *v7 )
        {
          while ( v32 != -4096 )
          {
            if ( v32 == -8192 && !v29 )
              v29 = v7;
            v30 = v27 & (v31 + v30);
            v7 = (__int64 *)(v28 + 16LL * v30);
            v32 = *v7;
            if ( a1 == *v7 )
              goto LABEL_15;
            ++v31;
          }
          if ( v29 )
            v7 = v29;
        }
        goto LABEL_15;
      }
LABEL_51:
      ++*(_DWORD *)(a2 + 16);
      BUG();
    }
  }
  else
  {
    ++*(_QWORD *)a2;
  }
  sub_2CF72E0(a2, 2 * v4);
  v19 = *(_DWORD *)(a2 + 24);
  if ( !v19 )
    goto LABEL_51;
  v20 = v19 - 1;
  v21 = *(_QWORD *)(a2 + 8);
  v22 = (v19 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
  v15 = *(_DWORD *)(a2 + 16) + 1;
  v7 = (__int64 *)(v21 + 16LL * v22);
  v23 = *v7;
  if ( a1 != *v7 )
  {
    v24 = 1;
    v25 = 0;
    while ( v23 != -4096 )
    {
      if ( !v25 && v23 == -8192 )
        v25 = v7;
      v22 = v20 & (v24 + v22);
      v7 = (__int64 *)(v21 + 16LL * v22);
      v23 = *v7;
      if ( a1 == *v7 )
        goto LABEL_15;
      ++v24;
    }
    if ( v25 )
      v7 = v25;
  }
LABEL_15:
  *(_DWORD *)(a2 + 16) = v15;
  if ( *v7 != -4096 )
    --*(_DWORD *)(a2 + 20);
  *v7 = a1;
  v16 = v33[0];
  *((_BYTE *)v7 + 8) = 0;
  v17 = *(_BYTE *)v16;
  if ( *(_BYTE *)v16 > 0x16u || ((0x40002FuLL >> v17) & 1) == 0 )
  {
    if ( v17 == 78 || v17 == 79 )
    {
      v18 = *(_QWORD *)(v16 - 32);
    }
    else
    {
      if ( v17 != 63 )
      {
        v12 = 1;
        *(_BYTE *)sub_2CF74C0(a2, v33) = 1;
        return v12;
      }
      v18 = *(_QWORD *)(v16 - 32LL * (*(_DWORD *)(v16 + 4) & 0x7FFFFFF));
    }
    v12 = sub_2CF7710(v18, a2);
    *(_BYTE *)sub_2CF74C0(a2, v33) = v12;
    return v12;
  }
  return 0;
}
