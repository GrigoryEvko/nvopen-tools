// Function: sub_1A3EEA0
// Address: 0x1a3eea0
//
unsigned __int64 __fastcall sub_1A3EEA0(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 *a4, __int64 a5, int a6)
{
  char *v7; // rsi
  unsigned __int64 result; // rax
  __int64 v9; // r12
  char *v10; // rdi
  char *v11; // rdx
  void *v12; // rdi
  __int64 v13; // [rsp+8h] [rbp-28h]
  __int64 v14; // [rsp+8h] [rbp-28h]

  *(_QWORD *)a1 = a2;
  v7 = (char *)(a1 + 56);
  *(_QWORD *)(a1 + 40) = a1 + 56;
  *(_QWORD *)(a1 + 48) = 0x800000000LL;
  result = *a4;
  *(_QWORD *)(a1 + 8) = a3;
  *(_QWORD *)(a1 + 16) = a4;
  *(_QWORD *)(a1 + 24) = a5;
  if ( *(_BYTE *)(result + 8) == 15 )
  {
    *(_QWORD *)(a1 + 32) = result;
    result = *(_QWORD *)(result + 24);
    v9 = *(_QWORD *)(result + 32);
    *(_DWORD *)(a1 + 120) = v9;
    if ( a5 )
    {
LABEL_3:
      result = *(unsigned int *)(a5 + 8);
      if ( !(_DWORD)result && (_DWORD)v9 )
      {
        if ( *(unsigned int *)(a5 + 12) < (unsigned __int64)(unsigned int)v9 )
        {
          v14 = a5;
          sub_16CD150(a5, (const void *)(a5 + 16), (unsigned int)v9, 8, a5, a6);
          a5 = v14;
          result = *(unsigned int *)(v14 + 8);
        }
        v12 = (void *)(*(_QWORD *)a5 + 8 * result);
        if ( v12 != (void *)(*(_QWORD *)a5 + 8LL * (unsigned int)v9) )
        {
          v13 = a5;
          result = (unsigned __int64)memset(v12, 0, 8 * ((unsigned int)v9 - result));
          a5 = v13;
        }
        *(_DWORD *)(a5 + 8) = v9;
      }
      return result;
    }
  }
  else
  {
    *(_QWORD *)(a1 + 32) = 0;
    v9 = *(_QWORD *)(result + 32);
    *(_DWORD *)(a1 + 120) = v9;
    if ( a5 )
      goto LABEL_3;
  }
  if ( (_DWORD)v9 )
  {
    v10 = (char *)(a1 + 56);
    if ( (unsigned int)v9 > 8uLL )
    {
      sub_16CD150(a1 + 40, v7, (unsigned int)v9, 8, a5, a6);
      v7 = *(char **)(a1 + 40);
      result = *(unsigned int *)(a1 + 48);
      v10 = &v7[8 * result];
    }
    v11 = &v7[8 * (unsigned int)v9];
    if ( v11 != v10 )
      result = (unsigned __int64)memset(v10, 0, v11 - v10);
    *(_DWORD *)(a1 + 48) = v9;
  }
  return result;
}
