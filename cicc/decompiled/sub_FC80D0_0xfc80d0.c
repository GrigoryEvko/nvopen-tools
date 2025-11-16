// Function: sub_FC80D0
// Address: 0xfc80d0
//
__int64 __fastcall sub_FC80D0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // rbx
  __int64 v7; // rdi
  unsigned int v8; // esi
  __int64 v9; // r8
  int v10; // r10d
  _QWORD *v11; // r13
  unsigned int v12; // ecx
  _QWORD *v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rsi
  __int64 *v16; // r13
  __int64 v18; // rax
  int v19; // esi
  int v20; // eax
  int v21; // ecx
  __int64 v22; // rdi
  unsigned int v23; // eax
  int v24; // edx
  __int64 v25; // rsi
  int v26; // r9d
  _QWORD *v27; // r8
  int v28; // eax
  int v29; // eax
  int v30; // eax
  __int64 v31; // rsi
  int v32; // r8d
  unsigned int v33; // r15d
  _QWORD *v34; // rdi
  __int64 v35; // rcx

  v5 = *(_QWORD *)(a1 + 24) + 16LL * *(unsigned int *)(a1 + 16);
  v6 = *(_QWORD *)v5;
  v7 = *(_QWORD *)v5 + 32LL;
  if ( !*(_BYTE *)(*(_QWORD *)v5 + 64LL) )
  {
    *(_QWORD *)(v6 + 40) = 0;
    v18 = 1;
    *(_QWORD *)(v6 + 48) = 0;
    *(_DWORD *)(v6 + 56) = 0;
    *(_BYTE *)(v6 + 64) = 1;
LABEL_10:
    *(_QWORD *)(v6 + 32) = v18;
    v19 = 0;
    goto LABEL_11;
  }
  v8 = *(_DWORD *)(v6 + 56);
  v9 = *(_QWORD *)(v6 + 40);
  if ( !v8 )
  {
    v18 = *(_QWORD *)(v6 + 32) + 1LL;
    goto LABEL_10;
  }
  v10 = 1;
  v11 = 0;
  v12 = (v8 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v13 = (_QWORD *)(v9 + 16LL * v12);
  v14 = *v13;
  if ( *v13 == a2 )
  {
LABEL_4:
    v15 = v13[1];
    v16 = v13 + 1;
    if ( v15 )
      sub_B91220((__int64)(v13 + 1), v15);
    goto LABEL_6;
  }
  while ( v14 != -4096 )
  {
    if ( !v11 && v14 == -8192 )
      v11 = v13;
    v12 = (v8 - 1) & (v10 + v12);
    v13 = (_QWORD *)(v9 + 16LL * v12);
    v14 = *v13;
    if ( *v13 == a2 )
      goto LABEL_4;
    ++v10;
  }
  if ( !v11 )
    v11 = v13;
  v28 = *(_DWORD *)(v6 + 48);
  ++*(_QWORD *)(v6 + 32);
  v24 = v28 + 1;
  if ( 4 * (v28 + 1) >= 3 * v8 )
  {
    v19 = 2 * v8;
LABEL_11:
    sub_FC7EB0(v7, v19);
    v20 = *(_DWORD *)(v6 + 56);
    if ( v20 )
    {
      v21 = v20 - 1;
      v22 = *(_QWORD *)(v6 + 40);
      v23 = (v20 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v24 = *(_DWORD *)(v6 + 48) + 1;
      v11 = (_QWORD *)(v22 + 16LL * v23);
      v25 = *v11;
      if ( *v11 != a2 )
      {
        v26 = 1;
        v27 = 0;
        while ( v25 != -4096 )
        {
          if ( !v27 && v25 == -8192 )
            v27 = v11;
          v23 = v21 & (v26 + v23);
          v11 = (_QWORD *)(v22 + 16LL * v23);
          v25 = *v11;
          if ( *v11 == a2 )
            goto LABEL_28;
          ++v26;
        }
        if ( v27 )
          v11 = v27;
      }
      goto LABEL_28;
    }
    goto LABEL_49;
  }
  if ( v8 - *(_DWORD *)(v6 + 52) - v24 <= v8 >> 3 )
  {
    sub_FC7EB0(v7, v8);
    v29 = *(_DWORD *)(v6 + 56);
    if ( v29 )
    {
      v30 = v29 - 1;
      v31 = *(_QWORD *)(v6 + 40);
      v32 = 1;
      v33 = v30 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v34 = 0;
      v24 = *(_DWORD *)(v6 + 48) + 1;
      v11 = (_QWORD *)(v31 + 16LL * v33);
      v35 = *v11;
      if ( *v11 != a2 )
      {
        while ( v35 != -4096 )
        {
          if ( v35 == -8192 && !v34 )
            v34 = v11;
          v33 = v30 & (v32 + v33);
          v11 = (_QWORD *)(v31 + 16LL * v33);
          v35 = *v11;
          if ( *v11 == a2 )
            goto LABEL_28;
          ++v32;
        }
        if ( v34 )
          v11 = v34;
      }
      goto LABEL_28;
    }
LABEL_49:
    ++*(_DWORD *)(v6 + 48);
    BUG();
  }
LABEL_28:
  *(_DWORD *)(v6 + 48) = v24;
  if ( *v11 != -4096 )
    --*(_DWORD *)(v6 + 52);
  *v11 = a2;
  v16 = v11 + 1;
  *v16 = 0;
LABEL_6:
  *v16 = a3;
  if ( a3 )
    sub_B96E90((__int64)v16, a3, 1);
  return a3;
}
