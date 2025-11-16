// Function: sub_1088350
// Address: 0x1088350
//
__int64 *__fastcall sub_1088350(__int64 a1)
{
  __int64 *result; // rax
  __int64 *v2; // r8
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // rsi
  __int64 *i; // r8
  __int64 v7; // rcx
  __int64 v8; // rsi

  result = *(__int64 **)(a1 + 48);
  v2 = *(__int64 **)(a1 + 56);
  if ( result != v2 )
  {
    v3 = 1;
    do
    {
      v4 = *result;
      v5 = *(_QWORD *)(*result + 88);
      if ( *(_BYTE *)(*(_QWORD *)(v5 + 64) + 20LL) != 5 )
      {
        *(_DWORD *)(v4 + 72) = v3;
        *(_DWORD *)(v5 + 12) = v3;
        *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v4 + 88) + 64LL) + 16LL) = v3++;
      }
      ++result;
    }
    while ( result != v2 );
    result = *(__int64 **)(a1 + 48);
    for ( i = *(__int64 **)(a1 + 56); i != result; ++v3 )
    {
      while ( 1 )
      {
        v7 = *result;
        v8 = *(_QWORD *)(*result + 88);
        if ( *(_BYTE *)(*(_QWORD *)(v8 + 64) + 20LL) == 5 )
          break;
        if ( i == ++result )
          return result;
      }
      *(_DWORD *)(v7 + 72) = v3;
      ++result;
      *(_DWORD *)(v8 + 12) = v3;
      *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v7 + 88) + 64LL) + 16LL) = v3;
    }
  }
  return result;
}
