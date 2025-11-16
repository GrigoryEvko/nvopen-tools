// Function: sub_1BC12F0
// Address: 0x1bc12f0
//
_QWORD *__fastcall sub_1BC12F0(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  int v8; // eax
  __int64 v9; // rcx
  __int64 v10; // rsi
  int v11; // edi
  unsigned int v12; // edx
  __int64 *v13; // rax
  __int64 v14; // rdx
  _QWORD *v15; // rax
  __int64 v16; // rcx
  unsigned __int64 v17; // r13
  char **v18; // rbx
  char *v19; // rax
  __int64 v20; // rdi
  _QWORD *v21; // rdx
  _QWORD *v22; // rax
  unsigned __int64 v23; // rdi
  unsigned __int64 v24; // rdi
  _QWORD *v26; // rdx
  _QWORD *v27; // rcx
  _QWORD *v28; // rax
  unsigned __int64 v29; // rsi
  int v30; // eax

  v8 = *(_DWORD *)(a1 + 24);
  if ( v8 )
  {
    v9 = *a2;
    v10 = *(_QWORD *)(a1 + 8);
    v11 = v8 - 1;
    v12 = (v8 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
    v13 = (__int64 *)(v10 + 16LL * v12);
    a5 = *v13;
    if ( *a2 == *v13 )
    {
LABEL_3:
      *v13 = -16;
      --*(_DWORD *)(a1 + 16);
      ++*(_DWORD *)(a1 + 20);
    }
    else
    {
      v30 = 1;
      while ( a5 != -8 )
      {
        a6 = v30 + 1;
        v12 = v11 & (v30 + v12);
        v13 = (__int64 *)(v10 + 16LL * v12);
        a5 = *v13;
        if ( v9 == *v13 )
          goto LABEL_3;
        v30 = a6;
      }
    }
  }
  v14 = *(_QWORD *)(a1 + 40);
  v15 = a2 + 5;
  if ( (_QWORD *)v14 != a2 + 5 )
  {
    v16 = v14 - (_QWORD)v15;
    v17 = 0xCCCCCCCCCCCCCCCDLL * ((v14 - (__int64)v15) >> 3);
    if ( v14 - (__int64)v15 <= 0 )
    {
      v15 = *(_QWORD **)(a1 + 40);
    }
    else
    {
      v18 = (char **)(a2 + 1);
      do
      {
        v19 = v18[4];
        v20 = (__int64)v18;
        v18 += 5;
        *(v18 - 6) = v19;
        sub_1BB9C60(v20, v18, v14, v16, a5, a6);
        --v17;
      }
      while ( v17 );
      v15 = *(_QWORD **)(a1 + 40);
    }
  }
  v21 = v15 - 5;
  v22 = v15 - 2;
  *(_QWORD *)(a1 + 40) = v21;
  v23 = *(v22 - 2);
  if ( (_QWORD *)v23 != v22 )
    _libc_free(v23);
  if ( *(_QWORD **)(a1 + 40) != a2 )
  {
    v24 = 0xCCCCCCCCCCCCCCCDLL * (((__int64)a2 - *(_QWORD *)(a1 + 32)) >> 3);
    if ( *(_DWORD *)(a1 + 16) )
    {
      v26 = *(_QWORD **)(a1 + 8);
      v27 = &v26[2 * *(unsigned int *)(a1 + 24)];
      if ( v26 != v27 )
      {
        while ( 1 )
        {
          v28 = v26;
          if ( *v26 != -16 && *v26 != -8 )
            break;
          v26 += 2;
          if ( v27 == v26 )
            return a2;
        }
        while ( v27 != v28 )
        {
          v29 = *((unsigned int *)v28 + 2);
          if ( v29 > v24 )
            *((_DWORD *)v28 + 2) = v29 - 1;
          v28 += 2;
          if ( v28 == v27 )
            break;
          while ( *v28 == -16 || *v28 == -8 )
          {
            v28 += 2;
            if ( v27 == v28 )
              return a2;
          }
        }
      }
    }
  }
  return a2;
}
