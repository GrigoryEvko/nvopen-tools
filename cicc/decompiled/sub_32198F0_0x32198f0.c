// Function: sub_32198F0
// Address: 0x32198f0
//
__int64 __fastcall sub_32198F0(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v8; // edx
  __int64 v9; // rsi
  int v10; // edi
  unsigned int v11; // eax
  __int64 *v12; // rdx
  __int64 v13; // rdx
  unsigned int v14; // eax
  __int64 v15; // rcx
  __int64 v16; // r12
  char **v17; // rbx
  char *v18; // rax
  __int64 v19; // rdi
  __int64 v20; // rax
  __int64 v21; // rax
  unsigned __int64 v22; // rdi
  __int64 result; // rax
  unsigned __int64 v24; // r13
  _QWORD *v25; // rcx
  _QWORD *v26; // rdx
  unsigned __int64 v27; // rsi
  int v28; // edx

  v8 = *(_DWORD *)(a1 + 24);
  v9 = *(_QWORD *)(a1 + 8);
  if ( v8 )
  {
    v10 = v8 - 1;
    v11 = (v8 - 1) & (((0xBF58476D1CE4E5B9LL * *a2) >> 31) ^ (484763065 * *(_DWORD *)a2));
    v12 = (__int64 *)(v9 + 16LL * v11);
    a5 = *v12;
    if ( *a2 == *v12 )
    {
LABEL_3:
      *v12 = -2;
      --*(_DWORD *)(a1 + 16);
      ++*(_DWORD *)(a1 + 20);
    }
    else
    {
      v28 = 1;
      while ( a5 != -1 )
      {
        a6 = (unsigned int)(v28 + 1);
        v11 = v10 & (v28 + v11);
        v12 = (__int64 *)(v9 + 16LL * v11);
        a5 = *v12;
        if ( *a2 == *v12 )
          goto LABEL_3;
        v28 = a6;
      }
    }
  }
  v13 = *(_QWORD *)(a1 + 32);
  v14 = *(_DWORD *)(a1 + 40);
  v15 = v13 + 56LL * v14 - (_QWORD)(a2 + 7);
  v16 = 0x6DB6DB6DB6DB6DB7LL * (v15 >> 3);
  if ( v15 > 0 )
  {
    v17 = (char **)(a2 + 1);
    do
    {
      v18 = v17[6];
      v19 = (__int64)v17;
      v17 += 7;
      *(v17 - 8) = v18;
      sub_32187E0(v19, v17, v13, v15, a5, a6);
      --v16;
    }
    while ( v16 );
    v14 = *(_DWORD *)(a1 + 40);
    v13 = *(_QWORD *)(a1 + 32);
  }
  v20 = v14 - 1;
  *(_DWORD *)(a1 + 40) = v20;
  v21 = v13 + 56 * v20;
  v22 = *(_QWORD *)(v21 + 8);
  if ( v22 != v21 + 24 )
  {
    _libc_free(v22);
    v13 = *(_QWORD *)(a1 + 32);
  }
  result = v13 + 56LL * *(unsigned int *)(a1 + 40);
  if ( a2 != (_QWORD *)result )
  {
    v24 = 0x6DB6DB6DB6DB6DB7LL * (((__int64)a2 - v13) >> 3);
    result = *(unsigned int *)(a1 + 16);
    if ( (_DWORD)result )
    {
      v25 = *(_QWORD **)(a1 + 8);
      v26 = &v25[2 * *(unsigned int *)(a1 + 24)];
      if ( v25 != v26 )
      {
        while ( 1 )
        {
          result = (__int64)v25;
          if ( *v25 <= 0xFFFFFFFFFFFFFFFDLL )
            break;
          v25 += 2;
          if ( v26 == v25 )
            return result;
        }
        if ( v25 != v26 )
        {
          do
          {
            v27 = *(unsigned int *)(result + 8);
            if ( v24 < v27 )
              *(_DWORD *)(result + 8) = v27 - 1;
            result += 16;
            if ( (_QWORD *)result == v26 )
              break;
            while ( *(_QWORD *)result > 0xFFFFFFFFFFFFFFFDLL )
            {
              result += 16;
              if ( v26 == (_QWORD *)result )
                return result;
            }
          }
          while ( v26 != (_QWORD *)result );
        }
      }
    }
  }
  return result;
}
