// Function: sub_27918D0
// Address: 0x27918d0
//
_QWORD *__fastcall sub_27918D0(__int64 a1, int a2, __int64 a3, __int64 a4)
{
  unsigned int v8; // esi
  __int64 v9; // r8
  _DWORD *v10; // rdx
  int v11; // r10d
  unsigned int v12; // edi
  int *v13; // rax
  int v14; // ecx
  __int64 v15; // rcx
  _QWORD *v16; // rdx
  _QWORD *result; // rax
  _QWORD *v18; // rsi
  int v19; // eax
  int v20; // ecx
  int v21; // eax
  int v22; // edi
  __int64 v23; // r10
  unsigned int v24; // eax
  int v25; // esi
  int v26; // r9d
  _DWORD *v27; // r8
  int v28; // eax
  int v29; // esi
  __int64 v30; // r8
  _DWORD *v31; // r9
  __int64 v32; // r15
  int v33; // edi
  int v34; // eax

  v8 = *(_DWORD *)(a1 + 24);
  if ( !v8 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_29;
  }
  v9 = *(_QWORD *)(a1 + 8);
  v10 = 0;
  v11 = 1;
  v12 = (v8 - 1) & (37 * a2);
  v13 = (int *)(v9 + 40LL * v12);
  v14 = *v13;
  if ( *v13 != a2 )
  {
    while ( v14 != -1 )
    {
      if ( v14 == -2 && !v10 )
        v10 = v13;
      v12 = (v8 - 1) & (v11 + v12);
      v13 = (int *)(v9 + 40LL * v12);
      v14 = *v13;
      if ( *v13 == a2 )
        goto LABEL_3;
      ++v11;
    }
    if ( !v10 )
      v10 = v13;
    v19 = *(_DWORD *)(a1 + 16);
    ++*(_QWORD *)a1;
    v20 = v19 + 1;
    if ( 4 * (v19 + 1) < 3 * v8 )
    {
      if ( v8 - *(_DWORD *)(a1 + 20) - v20 > v8 >> 3 )
      {
LABEL_25:
        *(_DWORD *)(a1 + 16) = v20;
        if ( *v10 != -1 )
          --*(_DWORD *)(a1 + 20);
        *v10 = a2;
        result = 0;
        v16 = v10 + 2;
        v15 = 0;
        *v16 = 0;
        v16[1] = 0;
        v16[2] = 0;
        v16[3] = 0;
        goto LABEL_4;
      }
      sub_27913C0(a1, v8);
      v28 = *(_DWORD *)(a1 + 24);
      if ( v28 )
      {
        v29 = v28 - 1;
        v30 = *(_QWORD *)(a1 + 8);
        v31 = 0;
        LODWORD(v32) = (v28 - 1) & (37 * a2);
        v20 = *(_DWORD *)(a1 + 16) + 1;
        v33 = 1;
        v10 = (_DWORD *)(v30 + 40LL * (unsigned int)v32);
        v34 = *v10;
        if ( *v10 != a2 )
        {
          while ( v34 != -1 )
          {
            if ( v34 == -2 && !v31 )
              v31 = v10;
            v32 = v29 & (unsigned int)(v32 + v33);
            v10 = (_DWORD *)(v30 + 40 * v32);
            v34 = *v10;
            if ( *v10 == a2 )
              goto LABEL_25;
            ++v33;
          }
          if ( v31 )
            v10 = v31;
        }
        goto LABEL_25;
      }
LABEL_52:
      ++*(_DWORD *)(a1 + 16);
      BUG();
    }
LABEL_29:
    sub_27913C0(a1, 2 * v8);
    v21 = *(_DWORD *)(a1 + 24);
    if ( v21 )
    {
      v22 = v21 - 1;
      v23 = *(_QWORD *)(a1 + 8);
      v24 = (v21 - 1) & (37 * a2);
      v20 = *(_DWORD *)(a1 + 16) + 1;
      v10 = (_DWORD *)(v23 + 40LL * v24);
      v25 = *v10;
      if ( *v10 != a2 )
      {
        v26 = 1;
        v27 = 0;
        while ( v25 != -1 )
        {
          if ( !v27 && v25 == -2 )
            v27 = v10;
          v24 = v22 & (v26 + v24);
          v10 = (_DWORD *)(v23 + 40LL * v24);
          v25 = *v10;
          if ( *v10 == a2 )
            goto LABEL_25;
          ++v26;
        }
        if ( v27 )
          v10 = v27;
      }
      goto LABEL_25;
    }
    goto LABEL_52;
  }
LABEL_3:
  v15 = *((_QWORD *)v13 + 1);
  v16 = v13 + 2;
  result = (_QWORD *)*((_QWORD *)v13 + 4);
LABEL_4:
  v18 = 0;
  while ( a3 != v15 || v16[1] != a4 )
  {
    if ( !result )
      return result;
    v15 = *result;
    v18 = v16;
    v16 = result;
    result = (_QWORD *)result[3];
  }
  if ( v18 )
  {
    v18[3] = result;
  }
  else if ( result )
  {
    *v16 = *result;
    v16[1] = result[1];
    v16[3] = result[3];
    result = (_QWORD *)result[2];
    v16[2] = result;
  }
  else
  {
    *v16 = 0;
    v16[1] = 0;
  }
  return result;
}
