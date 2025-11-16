// Function: sub_D5BC20
// Address: 0xd5bc20
//
__int64 *__fastcall sub_D5BC20(__int64 *a1, __int64 *a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rdi
  __int64 v7; // rsi
  __int64 *result; // rax

  v6 = *a1;
  v7 = *a2;
  if ( !*(_BYTE *)(v6 + 396) )
    return sub_C8CC70(v6 + 368, v7, (__int64)a3, a4, a5, a6);
  result = *(__int64 **)(v6 + 376);
  a4 = *(unsigned int *)(v6 + 388);
  a3 = &result[a4];
  if ( result == a3 )
  {
LABEL_7:
    if ( (unsigned int)a4 >= *(_DWORD *)(v6 + 384) )
      return sub_C8CC70(v6 + 368, v7, (__int64)a3, a4, a5, a6);
    *(_DWORD *)(v6 + 388) = a4 + 1;
    *a3 = v7;
    ++*(_QWORD *)(v6 + 368);
  }
  else
  {
    while ( v7 != *result )
    {
      if ( a3 == ++result )
        goto LABEL_7;
    }
  }
  return result;
}
