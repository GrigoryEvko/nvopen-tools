// Function: sub_22AD2D0
// Address: 0x22ad2d0
//
__int64 *__fastcall sub_22AD2D0(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *result; // rax

  if ( !*(_BYTE *)(a1 + 108) )
    return sub_C8CC70(a1 + 80, a2, (__int64)a3, a4, a5, a6);
  result = *(__int64 **)(a1 + 88);
  a4 = *(unsigned int *)(a1 + 100);
  a3 = &result[a4];
  if ( result == a3 )
  {
LABEL_7:
    if ( (unsigned int)a4 >= *(_DWORD *)(a1 + 96) )
      return sub_C8CC70(a1 + 80, a2, (__int64)a3, a4, a5, a6);
    *(_DWORD *)(a1 + 100) = a4 + 1;
    *a3 = a2;
    ++*(_QWORD *)(a1 + 80);
  }
  else
  {
    while ( a2 != *result )
    {
      if ( a3 == ++result )
        goto LABEL_7;
    }
  }
  return result;
}
