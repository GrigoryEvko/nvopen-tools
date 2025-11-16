// Function: sub_31F42C0
// Address: 0x31f42c0
//
unsigned __int64 __fastcall sub_31F42C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rcx
  unsigned __int64 result; // rax
  _QWORD *v9; // rsi
  __int64 v10; // r8
  unsigned __int64 v11; // rdx

  v7 = *(unsigned int *)(a1 + 8);
  result = *(_QWORD *)a1;
  v9 = (_QWORD *)(*(_QWORD *)a1 + 8 * v7);
  v10 = (8 * v7) >> 3;
  if ( (8 * v7) >> 5 )
  {
    v11 = result + 32 * ((8 * v7) >> 5);
    while ( a2 != *(_QWORD *)result )
    {
      if ( a2 == *(_QWORD *)(result + 8) )
      {
        result += 8LL;
        break;
      }
      if ( a2 == *(_QWORD *)(result + 16) )
      {
        result += 16LL;
        break;
      }
      if ( a2 == *(_QWORD *)(result + 24) )
      {
        result += 24LL;
        break;
      }
      result += 32LL;
      if ( result == v11 )
      {
        v10 = (__int64)((__int64)v9 - result) >> 3;
        goto LABEL_11;
      }
    }
LABEL_8:
    if ( v9 != (_QWORD *)result )
      return result;
    goto LABEL_14;
  }
LABEL_11:
  if ( v10 != 2 )
  {
    if ( v10 != 3 )
    {
      if ( v10 != 1 )
        goto LABEL_14;
      goto LABEL_21;
    }
    if ( a2 == *(_QWORD *)result )
      goto LABEL_8;
    result += 8LL;
  }
  if ( a2 == *(_QWORD *)result )
    goto LABEL_8;
  result += 8LL;
LABEL_21:
  if ( a2 == *(_QWORD *)result )
    goto LABEL_8;
LABEL_14:
  result = *(unsigned int *)(a1 + 12);
  if ( v7 + 1 > result )
  {
    sub_C8D5F0(a1, (const void *)(a1 + 16), v7 + 1, 8u, v10, a6);
    result = *(_QWORD *)a1;
    v9 = (_QWORD *)(*(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8));
  }
  *v9 = a2;
  ++*(_DWORD *)(a1 + 8);
  return result;
}
