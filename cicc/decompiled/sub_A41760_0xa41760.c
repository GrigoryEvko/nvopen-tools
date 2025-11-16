// Function: sub_A41760
// Address: 0xa41760
//
int *__fastcall sub_A41760(__int64 a1, __int64 a2)
{
  __int64 v3; // rcx
  unsigned int v5; // esi
  unsigned int v6; // edi
  unsigned int v7; // edx
  __int64 *v8; // rax
  __int64 v9; // r8
  int *result; // rax
  unsigned int v11; // eax
  int v12; // r13d
  int v13; // r11d
  _QWORD *v14; // r8
  unsigned int v15; // r9d
  _QWORD *v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  unsigned __int8 **v19; // r14
  unsigned __int8 **v20; // r13
  unsigned __int8 v21; // al
  int v22; // eax
  int v23; // esi
  __int64 v24; // rdi
  unsigned int v25; // ecx
  int v26; // edx
  __int64 v27; // rax
  int v28; // eax
  int v29; // r9d
  int v30; // eax
  int v31; // ecx
  __int64 v32; // rsi
  int v33; // r9d
  unsigned int v34; // r14d
  _QWORD *v35; // rdi
  __int64 v36; // rax
  __int64 v37; // rax
  int v38; // r10d
  _QWORD *v39; // r9

  v3 = *(_QWORD *)(a2 + 8);
  v5 = *(_DWORD *)(a2 + 24);
  if ( v5 )
  {
    v6 = v5 - 1;
    v7 = (v5 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
    v8 = (__int64 *)(v3 + 16LL * v7);
    v9 = *v8;
    if ( a1 == *v8 )
    {
LABEL_3:
      result = (int *)*((unsigned int *)v8 + 2);
      if ( (_DWORD)result )
        return result;
    }
    else
    {
      v28 = 1;
      while ( v9 != -4096 )
      {
        v29 = v28 + 1;
        v7 = v6 & (v28 + v7);
        v8 = (__int64 *)(v3 + 16LL * v7);
        v9 = *v8;
        if ( a1 == *v8 )
          goto LABEL_3;
        v28 = v29;
      }
    }
    if ( *(_BYTE *)a1 > 0x15u || (v11 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF) == 0 )
    {
      v12 = *(_DWORD *)(a2 + 16) + 1;
      goto LABEL_7;
    }
  }
  else if ( *(_BYTE *)a1 > 0x15u || (v11 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF) == 0 )
  {
    v12 = *(_DWORD *)(a2 + 16) + 1;
LABEL_26:
    ++*(_QWORD *)a2;
    v5 = 0;
    goto LABEL_27;
  }
  v18 = 4LL * v11;
  if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
  {
    v20 = *(unsigned __int8 ***)(a1 - 8);
    v19 = &v20[v18];
  }
  else
  {
    v19 = (unsigned __int8 **)a1;
    v20 = (unsigned __int8 **)(a1 - v18 * 8);
  }
  do
  {
    v21 = **v20;
    if ( v21 > 3u && v21 != 23 )
      sub_A41760(*v20, a2);
    v20 += 4;
  }
  while ( v19 != v20 );
  if ( *(_BYTE *)a1 == 5 && *(_WORD *)(a1 + 2) == 63 )
  {
    v37 = sub_AC3600(a1);
    sub_A41760(v37, a2);
  }
  v5 = *(_DWORD *)(a2 + 24);
  v3 = *(_QWORD *)(a2 + 8);
  v12 = *(_DWORD *)(a2 + 16) + 1;
  if ( !v5 )
    goto LABEL_26;
  v6 = v5 - 1;
LABEL_7:
  v13 = 1;
  v14 = 0;
  v15 = v6 & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
  v16 = (_QWORD *)(v3 + 16LL * v15);
  v17 = *v16;
  if ( a1 != *v16 )
  {
    while ( v17 != -4096 )
    {
      if ( v17 == -8192 && !v14 )
        v14 = v16;
      v15 = v6 & (v13 + v15);
      v16 = (_QWORD *)(v3 + 16LL * v15);
      v17 = *v16;
      if ( a1 == *v16 )
        goto LABEL_8;
      ++v13;
    }
    if ( !v14 )
      v14 = v16;
    ++*(_QWORD *)a2;
    if ( 4 * v12 < 3 * v5 )
    {
      v26 = v12;
      if ( v5 - (v12 + *(_DWORD *)(a2 + 20)) > v5 >> 3 )
      {
LABEL_29:
        *(_DWORD *)(a2 + 16) = v26;
        if ( *v14 != -4096 )
          --*(_DWORD *)(a2 + 20);
        *v14 = a1;
        result = (int *)(v14 + 1);
        *((_DWORD *)v14 + 2) = 0;
        *((_BYTE *)v14 + 12) = 0;
        goto LABEL_9;
      }
      sub_A41580(a2, v5);
      v30 = *(_DWORD *)(a2 + 24);
      if ( v30 )
      {
        v31 = v30 - 1;
        v32 = *(_QWORD *)(a2 + 8);
        v33 = 1;
        v34 = (v30 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
        v26 = *(_DWORD *)(a2 + 16) + 1;
        v35 = 0;
        v14 = (_QWORD *)(v32 + 16LL * v34);
        v36 = *v14;
        if ( a1 != *v14 )
        {
          while ( v36 != -4096 )
          {
            if ( !v35 && v36 == -8192 )
              v35 = v14;
            v34 = v31 & (v33 + v34);
            v14 = (_QWORD *)(v32 + 16LL * v34);
            v36 = *v14;
            if ( a1 == *v14 )
              goto LABEL_29;
            ++v33;
          }
          if ( v35 )
            v14 = v35;
        }
        goto LABEL_29;
      }
LABEL_67:
      ++*(_DWORD *)(a2 + 16);
      BUG();
    }
LABEL_27:
    sub_A41580(a2, 2 * v5);
    v22 = *(_DWORD *)(a2 + 24);
    if ( v22 )
    {
      v23 = v22 - 1;
      v24 = *(_QWORD *)(a2 + 8);
      v25 = (v22 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v26 = *(_DWORD *)(a2 + 16) + 1;
      v14 = (_QWORD *)(v24 + 16LL * v25);
      v27 = *v14;
      if ( a1 != *v14 )
      {
        v38 = 1;
        v39 = 0;
        while ( v27 != -4096 )
        {
          if ( !v39 && v27 == -8192 )
            v39 = v14;
          v25 = v23 & (v38 + v25);
          v14 = (_QWORD *)(v24 + 16LL * v25);
          v27 = *v14;
          if ( a1 == *v14 )
            goto LABEL_29;
          ++v38;
        }
        if ( v39 )
          v14 = v39;
      }
      goto LABEL_29;
    }
    goto LABEL_67;
  }
LABEL_8:
  result = (int *)(v16 + 1);
LABEL_9:
  *result = v12;
  return result;
}
