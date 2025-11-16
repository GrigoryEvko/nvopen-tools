// Function: sub_8E5930
// Address: 0x8e5930
//
unsigned __int8 *__fastcall sub_8E5930(unsigned __int8 *a1, __int64 a2)
{
  __int64 v2; // rbp
  __int64 v3; // r9
  unsigned __int8 *result; // rax
  unsigned __int8 *v5; // rdx
  __int64 v6[2]; // [rsp-10h] [rbp-10h] BYREF

  if ( *a1 != 66 )
    return a1;
  v6[1] = v2;
  v3 = a2;
  while ( 1 )
  {
    result = sub_8E5810(a1 + 1, v6, v3);
    if ( v6[0] <= 0 )
      break;
    v5 = &result[v6[0]];
    a1 = &result[v6[0]];
    while ( *result )
    {
      ++result;
      v6[0] = v5 - result;
      if ( result == v5 )
        goto LABEL_9;
    }
    if ( *(_DWORD *)(v3 + 24) )
      return result;
    ++*(_QWORD *)(v3 + 32);
    a1 = result;
    ++*(_QWORD *)(v3 + 48);
    *(_DWORD *)(v3 + 24) = 1;
LABEL_9:
    if ( *a1 != 66 )
      return a1;
  }
  if ( !*(_DWORD *)(v3 + 24) )
  {
    ++*(_QWORD *)(v3 + 32);
    ++*(_QWORD *)(v3 + 48);
    *(_DWORD *)(v3 + 24) = 1;
  }
  return result;
}
