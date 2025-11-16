// Function: sub_1B754F0
// Address: 0x1b754f0
//
__int64 __fastcall sub_1B754F0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // r12
  __int64 v7; // rdi
  unsigned int v8; // esi
  __int64 v9; // rcx
  unsigned int v10; // edx
  _QWORD *v11; // rbx
  __int64 v12; // rax
  __int64 v13; // rsi
  __int64 v14; // r12
  __int64 v16; // rax
  int v17; // esi
  int v18; // eax
  int v19; // ecx
  __int64 v20; // rdi
  unsigned int v21; // eax
  int v22; // edx
  __int64 v23; // rsi
  int v24; // r9d
  _QWORD *v25; // r8
  int v26; // r10d
  _QWORD *v27; // r9
  int v28; // eax
  int v29; // eax
  int v30; // eax
  __int64 v31; // rsi
  _QWORD *v32; // rdi
  unsigned int v33; // r15d
  int v34; // r8d
  __int64 v35; // rcx

  v5 = *(_QWORD *)(a1 + 24) + 16LL * *(unsigned int *)(a1 + 16);
  v6 = *(_QWORD *)v5;
  v7 = *(_QWORD *)v5 + 32LL;
  if ( !*(_BYTE *)(*(_QWORD *)v5 + 64LL) )
  {
    *(_BYTE *)(v6 + 64) = 1;
    v16 = 1;
    *(_QWORD *)(v6 + 40) = 0;
    *(_QWORD *)(v6 + 48) = 0;
    *(_DWORD *)(v6 + 56) = 0;
LABEL_10:
    *(_QWORD *)(v6 + 32) = v16;
    v17 = 0;
    goto LABEL_11;
  }
  v8 = *(_DWORD *)(v6 + 56);
  v9 = *(_QWORD *)(v6 + 40);
  if ( !v8 )
  {
    v16 = *(_QWORD *)(v6 + 32) + 1LL;
    goto LABEL_10;
  }
  v10 = (v8 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v11 = (_QWORD *)(v9 + 16LL * v10);
  v12 = *v11;
  if ( *v11 == a2 )
  {
LABEL_4:
    v13 = v11[1];
    v14 = (__int64)(v11 + 1);
    if ( v13 )
      sub_161E7C0((__int64)(v11 + 1), v13);
    goto LABEL_6;
  }
  v26 = 1;
  v27 = 0;
  while ( v12 != -4 )
  {
    if ( !v27 && v12 == -8 )
      v27 = v11;
    v10 = (v8 - 1) & (v26 + v10);
    v11 = (_QWORD *)(v9 + 16LL * v10);
    v12 = *v11;
    if ( *v11 == a2 )
      goto LABEL_4;
    ++v26;
  }
  v28 = *(_DWORD *)(v6 + 48);
  if ( v27 )
    v11 = v27;
  ++*(_QWORD *)(v6 + 32);
  v22 = v28 + 1;
  if ( 4 * (v28 + 1) >= 3 * v8 )
  {
    v17 = 2 * v8;
LABEL_11:
    sub_1671930(v7, v17);
    v18 = *(_DWORD *)(v6 + 56);
    if ( v18 )
    {
      v19 = v18 - 1;
      v20 = *(_QWORD *)(v6 + 40);
      v21 = (v18 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v22 = *(_DWORD *)(v6 + 48) + 1;
      v11 = (_QWORD *)(v20 + 16LL * v21);
      v23 = *v11;
      if ( *v11 != a2 )
      {
        v24 = 1;
        v25 = 0;
        while ( v23 != -4 )
        {
          if ( !v25 && v23 == -8 )
            v25 = v11;
          v21 = v19 & (v24 + v21);
          v11 = (_QWORD *)(v20 + 16LL * v21);
          v23 = *v11;
          if ( *v11 == a2 )
            goto LABEL_24;
          ++v24;
        }
        if ( v25 )
          v11 = v25;
      }
      goto LABEL_24;
    }
    goto LABEL_50;
  }
  if ( v8 - *(_DWORD *)(v6 + 52) - v22 <= v8 >> 3 )
  {
    sub_1671930(v7, v8);
    v29 = *(_DWORD *)(v6 + 56);
    if ( v29 )
    {
      v30 = v29 - 1;
      v31 = *(_QWORD *)(v6 + 40);
      v32 = 0;
      v33 = v30 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v34 = 1;
      v22 = *(_DWORD *)(v6 + 48) + 1;
      v11 = (_QWORD *)(v31 + 16LL * v33);
      v35 = *v11;
      if ( *v11 != a2 )
      {
        while ( v35 != -4 )
        {
          if ( !v32 && v35 == -8 )
            v32 = v11;
          v33 = v30 & (v34 + v33);
          v11 = (_QWORD *)(v31 + 16LL * v33);
          v35 = *v11;
          if ( *v11 == a2 )
            goto LABEL_24;
          ++v34;
        }
        if ( v32 )
          v11 = v32;
      }
      goto LABEL_24;
    }
LABEL_50:
    ++*(_DWORD *)(v6 + 48);
    BUG();
  }
LABEL_24:
  *(_DWORD *)(v6 + 48) = v22;
  if ( *v11 != -4 )
    --*(_DWORD *)(v6 + 52);
  *v11 = a2;
  v14 = (__int64)(v11 + 1);
  v11[1] = 0;
LABEL_6:
  v11[1] = a3;
  if ( a3 )
    sub_1623A60(v14, a3, 2);
  return a3;
}
