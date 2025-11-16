// Function: sub_731E00
// Address: 0x731e00
//
_BOOL8 __fastcall sub_731E00(__int64 a1)
{
  _BOOL8 result; // rax
  __int64 v2; // rdx
  __int64 i; // rax
  __int64 v4; // rcx

  result = 0;
  if ( *(_BYTE *)(a1 + 24) == 20 )
  {
    v2 = *(_QWORD *)(a1 + 56);
    if ( v2 )
    {
      if ( (*(_BYTE *)(v2 + 89) & 4) != 0 )
      {
        for ( i = *(_QWORD *)(v2 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
          ;
        v4 = *(_QWORD *)(i + 168);
        result = 0;
        if ( !*(_QWORD *)(v4 + 40) )
          return (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v2 + 40) + 32LL) + 177LL) & 0x20) != 0;
      }
    }
  }
  return result;
}
