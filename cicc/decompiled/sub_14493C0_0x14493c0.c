// Function: sub_14493C0
// Address: 0x14493c0
//
_QWORD *__fastcall sub_14493C0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rax
  _QWORD *v7; // r13
  unsigned int v8; // esi
  __int64 v9; // r9
  _QWORD *v10; // r8
  unsigned int v11; // edi
  _QWORD *v12; // rax
  __int64 v13; // rcx
  bool (__fastcall *v14)(__int64, _QWORD *); // rax
  int v16; // eax
  int v17; // esi
  unsigned int v18; // eax
  _QWORD *v19; // rdx
  __int64 v20; // rdi
  int v21; // r11d
  int v22; // eax
  int v23; // eax
  int v24; // eax
  __int64 v25; // rdi
  unsigned int v26; // r14d
  __int64 v27; // rsi
  int v28; // r10d

  if ( sub_1443EB0(a1, a2, a3) )
    return 0;
  v6 = sub_22077B0(112);
  v7 = (_QWORD *)v6;
  if ( v6 )
    sub_1444050(v6, a2, a3, a1, *(_QWORD *)(a1 + 8), 0);
  v8 = *(_DWORD *)(a1 + 64);
  v9 = a1 + 40;
  if ( !v8 )
  {
    ++*(_QWORD *)(a1 + 40);
    goto LABEL_12;
  }
  v10 = *(_QWORD **)(a1 + 48);
  v11 = (v8 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v12 = &v10[2 * v11];
  v13 = *v12;
  if ( a2 != *v12 )
  {
    v21 = 1;
    v19 = 0;
    while ( v13 != -8 )
    {
      if ( v19 || v13 != -16 )
        v12 = v19;
      v11 = (v8 - 1) & (v21 + v11);
      v13 = v10[2 * v11];
      if ( a2 == v13 )
        goto LABEL_6;
      ++v21;
      v19 = v12;
      v12 = &v10[2 * v11];
    }
    if ( !v19 )
      v19 = v12;
    v22 = *(_DWORD *)(a1 + 56);
    ++*(_QWORD *)(a1 + 40);
    v13 = (unsigned int)(v22 + 1);
    if ( 4 * (int)v13 < 3 * v8 )
    {
      if ( v8 - *(_DWORD *)(a1 + 60) - (unsigned int)v13 > v8 >> 3 )
      {
LABEL_14:
        *(_DWORD *)(a1 + 56) = v13;
        if ( *v19 != -8 )
          --*(_DWORD *)(a1 + 60);
        *v19 = a2;
        v19[1] = v7;
        goto LABEL_6;
      }
      sub_1448190(a1 + 40, v8);
      v23 = *(_DWORD *)(a1 + 64);
      if ( v23 )
      {
        v24 = v23 - 1;
        v25 = *(_QWORD *)(a1 + 48);
        v10 = 0;
        v26 = v24 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v9 = 1;
        v13 = (unsigned int)(*(_DWORD *)(a1 + 56) + 1);
        v19 = (_QWORD *)(v25 + 16LL * v26);
        v27 = *v19;
        if ( a2 != *v19 )
        {
          while ( v27 != -8 )
          {
            if ( v27 == -16 && !v10 )
              v10 = v19;
            v26 = v24 & (v9 + v26);
            v19 = (_QWORD *)(v25 + 16LL * v26);
            v27 = *v19;
            if ( a2 == *v19 )
              goto LABEL_14;
            v9 = (unsigned int)(v9 + 1);
          }
          if ( v10 )
            v19 = v10;
        }
        goto LABEL_14;
      }
LABEL_49:
      ++*(_DWORD *)(a1 + 56);
      BUG();
    }
LABEL_12:
    sub_1448190(a1 + 40, 2 * v8);
    v16 = *(_DWORD *)(a1 + 64);
    if ( v16 )
    {
      v17 = v16 - 1;
      v10 = *(_QWORD **)(a1 + 48);
      v18 = (v16 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v13 = (unsigned int)(*(_DWORD *)(a1 + 56) + 1);
      v19 = &v10[2 * v18];
      v20 = *v19;
      if ( a2 != *v19 )
      {
        v28 = 1;
        v9 = 0;
        while ( v20 != -8 )
        {
          if ( v20 == -16 && !v9 )
            v9 = (__int64)v19;
          v18 = v17 & (v28 + v18);
          v19 = &v10[2 * v18];
          v20 = *v19;
          if ( a2 == *v19 )
            goto LABEL_14;
          ++v28;
        }
        if ( v9 )
          v19 = (_QWORD *)v9;
      }
      goto LABEL_14;
    }
    goto LABEL_49;
  }
LABEL_6:
  v14 = *(bool (__fastcall **)(__int64, _QWORD *))(*(_QWORD *)a1 + 16LL);
  if ( v14 == sub_14439D0 )
    sub_1443980(v7);
  else
    ((void (__fastcall *)(__int64, _QWORD *, bool (__fastcall *)(__int64, _QWORD *), __int64, _QWORD *, __int64))v14)(
      a1,
      v7,
      sub_14439D0,
      v13,
      v10,
      v9);
  return v7;
}
