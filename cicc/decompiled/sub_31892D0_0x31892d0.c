// Function: sub_31892D0
// Address: 0x31892d0
//
_QWORD *__fastcall sub_31892D0(__int64 a1, __int64 a2)
{
  __int64 v4; // rdi
  unsigned int v5; // esi
  __int64 v6; // rcx
  int v7; // r11d
  _QWORD *v8; // rbx
  unsigned int v9; // edx
  _QWORD *v10; // rax
  __int64 v11; // r9
  _QWORD *v12; // r14
  int v14; // eax
  int v15; // edx
  __int64 v16; // rax
  __int64 v17; // rdi
  int v18; // eax
  int v19; // ecx
  __int64 v20; // rdi
  unsigned int v21; // eax
  __int64 v22; // rsi
  int v23; // r9d
  _QWORD *v24; // r8
  int v25; // eax
  int v26; // eax
  __int64 v27; // rsi
  _QWORD *v28; // rdi
  unsigned int v29; // r14d
  int v30; // r8d
  __int64 v31; // rcx

  v4 = a1 + 88;
  v5 = *(_DWORD *)(a1 + 112);
  if ( v5 )
  {
    v6 = *(_QWORD *)(a1 + 96);
    v7 = 1;
    v8 = 0;
    v9 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v10 = (_QWORD *)(v6 + 16LL * v9);
    v11 = *v10;
    if ( a2 == *v10 )
      return (_QWORD *)v10[1];
    while ( v11 != -4096 )
    {
      if ( !v8 && v11 == -8192 )
        v8 = v10;
      v9 = (v5 - 1) & (v7 + v9);
      v10 = (_QWORD *)(v6 + 16LL * v9);
      v11 = *v10;
      if ( a2 == *v10 )
        return (_QWORD *)v10[1];
      ++v7;
    }
    if ( !v8 )
      v8 = v10;
    v14 = *(_DWORD *)(a1 + 104);
    ++*(_QWORD *)(a1 + 88);
    v15 = v14 + 1;
    if ( 4 * (v14 + 1) < 3 * v5 )
    {
      if ( v5 - *(_DWORD *)(a1 + 108) - v15 > v5 >> 3 )
        goto LABEL_15;
      sub_3187CD0(v4, v5);
      v25 = *(_DWORD *)(a1 + 112);
      if ( v25 )
      {
        v26 = v25 - 1;
        v27 = *(_QWORD *)(a1 + 96);
        v28 = 0;
        v29 = v26 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v30 = 1;
        v15 = *(_DWORD *)(a1 + 104) + 1;
        v8 = (_QWORD *)(v27 + 16LL * v29);
        v31 = *v8;
        if ( a2 != *v8 )
        {
          while ( v31 != -4096 )
          {
            if ( v31 == -8192 && !v28 )
              v28 = v8;
            v29 = v26 & (v30 + v29);
            v8 = (_QWORD *)(v27 + 16LL * v29);
            v31 = *v8;
            if ( a2 == *v8 )
              goto LABEL_15;
            ++v30;
          }
          if ( v28 )
            v8 = v28;
        }
        goto LABEL_15;
      }
LABEL_45:
      ++*(_DWORD *)(a1 + 104);
      BUG();
    }
  }
  else
  {
    ++*(_QWORD *)(a1 + 88);
  }
  sub_3187CD0(v4, 2 * v5);
  v18 = *(_DWORD *)(a1 + 112);
  if ( !v18 )
    goto LABEL_45;
  v19 = v18 - 1;
  v20 = *(_QWORD *)(a1 + 96);
  v21 = (v18 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v15 = *(_DWORD *)(a1 + 104) + 1;
  v8 = (_QWORD *)(v20 + 16LL * v21);
  v22 = *v8;
  if ( a2 != *v8 )
  {
    v23 = 1;
    v24 = 0;
    while ( v22 != -4096 )
    {
      if ( !v24 && v22 == -8192 )
        v24 = v8;
      v21 = v19 & (v23 + v21);
      v8 = (_QWORD *)(v20 + 16LL * v21);
      v22 = *v8;
      if ( a2 == *v8 )
        goto LABEL_15;
      ++v23;
    }
    if ( v24 )
      v8 = v24;
  }
LABEL_15:
  *(_DWORD *)(a1 + 104) = v15;
  if ( *v8 != -4096 )
    --*(_DWORD *)(a1 + 108);
  *v8 = a2;
  v8[1] = 0;
  v16 = sub_22077B0(0x20u);
  v12 = (_QWORD *)v16;
  if ( v16 )
  {
    sub_318EB10(v16, 1, a2, a1);
    *v12 = &unk_4A32EB0;
  }
  v17 = v8[1];
  v8[1] = v12;
  if ( !v17 )
    return v12;
  (*(void (__fastcall **)(__int64))(*(_QWORD *)v17 + 8LL))(v17);
  return (_QWORD *)v8[1];
}
