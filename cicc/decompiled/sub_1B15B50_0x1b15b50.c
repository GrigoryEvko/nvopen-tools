// Function: sub_1B15B50
// Address: 0x1b15b50
//
unsigned __int64 __fastcall sub_1B15B50(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v8; // rdx
  __int64 v9; // rax
  unsigned __int64 result; // rax
  const void *v11; // rsi
  char *v12; // r13
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // rdx
  int v15; // r14d
  __int64 v16; // r15
  char *v17; // rsi
  __int64 v18; // rdx

  v8 = *(_QWORD *)(a1 + 16);
  v9 = *(_QWORD *)(a2 + 16);
  if ( v8 != v9 )
  {
    if ( v8 != -8 && v8 != 0 && v8 != -16 )
    {
      sub_1649B30((_QWORD *)a1);
      v9 = *(_QWORD *)(a2 + 16);
    }
    *(_QWORD *)(a1 + 16) = v9;
    if ( v9 != 0 && v9 != -8 && v9 != -16 )
      sub_1649AC0((unsigned __int64 *)a1, *(_QWORD *)a2 & 0xFFFFFFFFFFFFFFF8LL);
  }
  *(_DWORD *)(a1 + 24) = *(_DWORD *)(a2 + 24);
  *(_QWORD *)(a1 + 32) = *(_QWORD *)(a2 + 32);
  *(_QWORD *)(a1 + 40) = *(_QWORD *)(a2 + 40);
  result = a2 + 48;
  if ( a1 + 48 != a2 + 48 )
  {
    v11 = *(const void **)(a2 + 48);
    v12 = (char *)(a2 + 64);
    if ( v11 == (const void *)(a2 + 64) )
    {
      v14 = *(unsigned int *)(a2 + 56);
      result = *(unsigned int *)(a1 + 56);
      v15 = *(_DWORD *)(a2 + 56);
      if ( v14 <= result )
      {
        if ( *(_DWORD *)(a2 + 56) )
          result = (unsigned __int64)memmove(*(void **)(a1 + 48), v11, 8 * v14);
      }
      else
      {
        if ( v14 > *(unsigned int *)(a1 + 60) )
        {
          *(_DWORD *)(a1 + 56) = 0;
          sub_16CD150(a1 + 48, (const void *)(a1 + 64), v14, 8, a5, a6);
          v12 = *(char **)(a2 + 48);
          v14 = *(unsigned int *)(a2 + 56);
          result = 0;
          v17 = v12;
        }
        else
        {
          v16 = 8 * result;
          v17 = (char *)(a2 + 64);
          if ( *(_DWORD *)(a1 + 56) )
          {
            memmove(*(void **)(a1 + 48), v17, 8 * result);
            v12 = *(char **)(a2 + 48);
            v14 = *(unsigned int *)(a2 + 56);
            result = v16;
            v17 = &v12[v16];
          }
        }
        v18 = 8 * v14;
        if ( v17 != &v12[v18] )
          result = (unsigned __int64)memcpy((void *)(result + *(_QWORD *)(a1 + 48)), v17, v18 - result);
      }
      *(_DWORD *)(a1 + 56) = v15;
      *(_DWORD *)(a2 + 56) = 0;
    }
    else
    {
      v13 = *(_QWORD *)(a1 + 48);
      if ( v13 != a1 + 64 )
      {
        _libc_free(v13);
        v11 = *(const void **)(a2 + 48);
      }
      *(_QWORD *)(a1 + 48) = v11;
      *(_DWORD *)(a1 + 56) = *(_DWORD *)(a2 + 56);
      result = *(unsigned int *)(a2 + 60);
      *(_DWORD *)(a1 + 60) = result;
      *(_QWORD *)(a2 + 48) = v12;
      *(_QWORD *)(a2 + 56) = 0;
    }
  }
  return result;
}
