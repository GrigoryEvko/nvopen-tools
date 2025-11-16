// Function: sub_2A64130
// Address: 0x2a64130
//
__int64 *__fastcall sub_2A64130(__int64 *a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rdi
  __int64 *result; // rax

  v6 = *a1;
  if ( !*(_BYTE *)(v6 + 708) )
    return sub_C8CC70(v6 + 680, a2, (__int64)a3, a4, a5, a6);
  result = *(__int64 **)(v6 + 688);
  a4 = *(unsigned int *)(v6 + 700);
  a3 = &result[a4];
  if ( result == a3 )
  {
LABEL_7:
    if ( (unsigned int)a4 >= *(_DWORD *)(v6 + 696) )
      return sub_C8CC70(v6 + 680, a2, (__int64)a3, a4, a5, a6);
    *(_DWORD *)(v6 + 700) = a4 + 1;
    *a3 = a2;
    ++*(_QWORD *)(v6 + 680);
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
