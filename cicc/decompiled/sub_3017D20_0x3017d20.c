// Function: sub_3017D20
// Address: 0x3017d20
//
_QWORD *__fastcall sub_3017D20(__int64 a1, int a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r10
  unsigned int v9; // esi
  __int64 v10; // r8
  unsigned int v11; // eax
  unsigned int v12; // r15d
  int v13; // r11d
  _QWORD *v14; // rdx
  int v15; // r15d
  unsigned int v16; // edi
  _QWORD *v17; // rax
  __int64 v18; // rcx
  _QWORD *result; // rax
  int v20; // eax
  int v21; // ecx
  int v22; // eax
  int v23; // esi
  __int64 v24; // r8
  unsigned int v25; // eax
  __int64 v26; // rdi
  int v27; // r10d
  _QWORD *v28; // r9
  int v29; // eax
  int v30; // eax
  __int64 v31; // rdi
  int v32; // r9d
  unsigned int v33; // r15d
  _QWORD *v34; // r8
  __int64 v35; // rsi

  v4 = a1 + 96;
  v9 = *(_DWORD *)(a1 + 120);
  if ( !v9 )
  {
    ++*(_QWORD *)(a1 + 96);
    goto LABEL_19;
  }
  v10 = *(_QWORD *)(a1 + 104);
  v11 = (unsigned int)a3 >> 9;
  v12 = (unsigned int)a3 >> 4;
  v13 = 1;
  v14 = 0;
  v15 = v11 ^ v12;
  v16 = (v9 - 1) & v15;
  v17 = (_QWORD *)(v10 + 24LL * v16);
  v18 = *v17;
  if ( *v17 == a3 )
  {
LABEL_3:
    result = v17 + 1;
    goto LABEL_4;
  }
  while ( v18 != -4096 )
  {
    if ( !v14 && v18 == -8192 )
      v14 = v17;
    v16 = (v9 - 1) & (v13 + v16);
    v17 = (_QWORD *)(v10 + 24LL * v16);
    v18 = *v17;
    if ( *v17 == a3 )
      goto LABEL_3;
    ++v13;
  }
  if ( !v14 )
    v14 = v17;
  v20 = *(_DWORD *)(a1 + 112);
  ++*(_QWORD *)(a1 + 96);
  v21 = v20 + 1;
  if ( 4 * (v20 + 1) >= 3 * v9 )
  {
LABEL_19:
    sub_3017690(v4, 2 * v9);
    v22 = *(_DWORD *)(a1 + 120);
    if ( v22 )
    {
      v23 = v22 - 1;
      v24 = *(_QWORD *)(a1 + 104);
      v25 = (v22 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
      v21 = *(_DWORD *)(a1 + 112) + 1;
      v14 = (_QWORD *)(v24 + 24LL * v25);
      v26 = *v14;
      if ( *v14 != a3 )
      {
        v27 = 1;
        v28 = 0;
        while ( v26 != -4096 )
        {
          if ( !v28 && v26 == -8192 )
            v28 = v14;
          v25 = v23 & (v27 + v25);
          v14 = (_QWORD *)(v24 + 24LL * v25);
          v26 = *v14;
          if ( *v14 == a3 )
            goto LABEL_15;
          ++v27;
        }
        if ( v28 )
          v14 = v28;
      }
      goto LABEL_15;
    }
    goto LABEL_42;
  }
  if ( v9 - *(_DWORD *)(a1 + 116) - v21 <= v9 >> 3 )
  {
    sub_3017690(v4, v9);
    v29 = *(_DWORD *)(a1 + 120);
    if ( v29 )
    {
      v30 = v29 - 1;
      v31 = *(_QWORD *)(a1 + 104);
      v32 = 1;
      v33 = v30 & v15;
      v34 = 0;
      v21 = *(_DWORD *)(a1 + 112) + 1;
      v14 = (_QWORD *)(v31 + 24LL * v33);
      v35 = *v14;
      if ( *v14 != a3 )
      {
        while ( v35 != -4096 )
        {
          if ( !v34 && v35 == -8192 )
            v34 = v14;
          v33 = v30 & (v32 + v33);
          v14 = (_QWORD *)(v31 + 24LL * v33);
          v35 = *v14;
          if ( *v14 == a3 )
            goto LABEL_15;
          ++v32;
        }
        if ( v34 )
          v14 = v34;
      }
      goto LABEL_15;
    }
LABEL_42:
    ++*(_DWORD *)(a1 + 112);
    BUG();
  }
LABEL_15:
  *(_DWORD *)(a1 + 112) = v21;
  if ( *v14 != -4096 )
    --*(_DWORD *)(a1 + 116);
  *v14 = a3;
  result = v14 + 1;
  *((_DWORD *)v14 + 2) = 0;
  v14[2] = 0;
LABEL_4:
  *(_DWORD *)result = a2;
  result[1] = a4;
  return result;
}
