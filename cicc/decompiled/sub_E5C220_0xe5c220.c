// Function: sub_E5C220
// Address: 0xe5c220
//
__int64 *__fastcall sub_E5C220(__int64 a1, __int64 a2)
{
  __int64 *result; // rax
  __int64 v3; // rbx
  __int64 v4; // rsi
  __int64 v5; // r12

  result = (__int64 *)*(unsigned __int8 *)(a2 + 48);
  if ( ((unsigned __int8)result & 4) == 0 )
  {
    *(_BYTE *)(a2 + 48) = (unsigned __int8)result | 4;
    result = *(__int64 **)(a2 + 8);
    v3 = *result;
    if ( *result )
    {
      v4 = 0;
      v5 = 0;
      while ( 1 )
      {
        *(_QWORD *)(v3 + 16) = v5;
        if ( *(_DWORD *)(a1 + 368) )
        {
          if ( (*(_BYTE *)(v3 + 29) & 1) != 0 )
          {
            sub_E5C140(a1, v4, v3);
            v5 = *(_QWORD *)(v3 + 16);
          }
        }
        v4 = v3;
        v5 += sub_E5BD20((__int64 *)a1, v3);
        result = *(__int64 **)v3;
        if ( !*(_QWORD *)v3 )
          break;
        v3 = *(_QWORD *)v3;
      }
    }
  }
  return result;
}
