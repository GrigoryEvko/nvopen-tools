// Function: sub_2F69690
// Address: 0x2f69690
//
__int64 __fastcall sub_2F69690(__int64 a1, int a2)
{
  __int64 v4; // rdi
  unsigned int v5; // esi
  int v6; // r14d
  _DWORD *v7; // rdx
  __int64 v8; // r8
  unsigned int v9; // ecx
  int *v10; // rax
  int v11; // r10d
  _QWORD *v12; // rdx
  unsigned __int64 v13; // rax
  unsigned int v14; // r8d
  int v16; // eax
  int v17; // ecx
  int v18; // eax
  int v19; // esi
  __int64 v20; // r8
  unsigned int v21; // eax
  int v22; // edi
  int v23; // r10d
  _DWORD *v24; // r9
  int v25; // eax
  int v26; // eax
  __int64 v27; // rdi
  _DWORD *v28; // r8
  unsigned int v29; // r13d
  int v30; // r9d
  int v31; // esi

  v4 = a1 + 920;
  v5 = *(_DWORD *)(a1 + 944);
  if ( !v5 )
  {
    ++*(_QWORD *)(a1 + 920);
    goto LABEL_21;
  }
  v6 = 1;
  v7 = 0;
  v8 = *(_QWORD *)(a1 + 928);
  v9 = (v5 - 1) & (37 * a2);
  v10 = (int *)(v8 + 16LL * v9);
  v11 = *v10;
  if ( a2 == *v10 )
  {
LABEL_3:
    v12 = v10 + 2;
    v13 = *((_QWORD *)v10 + 1);
    goto LABEL_4;
  }
  while ( v11 != -1 )
  {
    if ( v11 == -2 && !v7 )
      v7 = v10;
    v9 = (v5 - 1) & (v6 + v9);
    v10 = (int *)(v8 + 16LL * v9);
    v11 = *v10;
    if ( a2 == *v10 )
      goto LABEL_3;
    ++v6;
  }
  if ( !v7 )
    v7 = v10;
  v16 = *(_DWORD *)(a1 + 936);
  ++*(_QWORD *)(a1 + 920);
  v17 = v16 + 1;
  if ( 4 * (v16 + 1) >= 3 * v5 )
  {
LABEL_21:
    sub_2F694B0(v4, 2 * v5);
    v18 = *(_DWORD *)(a1 + 944);
    if ( v18 )
    {
      v19 = v18 - 1;
      v20 = *(_QWORD *)(a1 + 928);
      v21 = (v18 - 1) & (37 * a2);
      v17 = *(_DWORD *)(a1 + 936) + 1;
      v7 = (_DWORD *)(v20 + 16LL * v21);
      v22 = *v7;
      if ( a2 != *v7 )
      {
        v23 = 1;
        v24 = 0;
        while ( v22 != -1 )
        {
          if ( !v24 && v22 == -2 )
            v24 = v7;
          v21 = v19 & (v23 + v21);
          v7 = (_DWORD *)(v20 + 16LL * v21);
          v22 = *v7;
          if ( a2 == *v7 )
            goto LABEL_17;
          ++v23;
        }
        if ( v24 )
          v7 = v24;
      }
      goto LABEL_17;
    }
    goto LABEL_44;
  }
  if ( v5 - *(_DWORD *)(a1 + 940) - v17 <= v5 >> 3 )
  {
    sub_2F694B0(v4, v5);
    v25 = *(_DWORD *)(a1 + 944);
    if ( v25 )
    {
      v26 = v25 - 1;
      v27 = *(_QWORD *)(a1 + 928);
      v28 = 0;
      v29 = v26 & (37 * a2);
      v30 = 1;
      v17 = *(_DWORD *)(a1 + 936) + 1;
      v7 = (_DWORD *)(v27 + 16LL * v29);
      v31 = *v7;
      if ( a2 != *v7 )
      {
        while ( v31 != -1 )
        {
          if ( v31 == -2 && !v28 )
            v28 = v7;
          v29 = v26 & (v30 + v29);
          v7 = (_DWORD *)(v27 + 16LL * v29);
          v31 = *v7;
          if ( a2 == *v7 )
            goto LABEL_17;
          ++v30;
        }
        if ( v28 )
          v7 = v28;
      }
      goto LABEL_17;
    }
LABEL_44:
    ++*(_DWORD *)(a1 + 936);
    BUG();
  }
LABEL_17:
  *(_DWORD *)(a1 + 936) = v17;
  if ( *v7 != -1 )
    --*(_DWORD *)(a1 + 940);
  *v7 = a2;
  v13 = 0;
  v12 = v7 + 2;
  *v12 = 0;
LABEL_4:
  v14 = 1;
  if ( (unsigned int)dword_5024808 > v13 )
  {
    v14 = 0;
    *v12 = v13 + 1;
  }
  return v14;
}
