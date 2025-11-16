// Function: sub_5E9060
// Address: 0x5e9060
//
_BOOL8 __fastcall sub_5E9060(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 i; // rbx
  __int64 j; // r12

  for ( i = *(_QWORD *)(a1 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  for ( j = *(_QWORD *)(*(_QWORD *)(a2 + 88) + 152LL); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
    ;
  return (i == j || (unsigned int)sub_8D97D0(i, j, 0, a4, a5))
      && (*(_BYTE *)(*(_QWORD *)(i + 168) + 17LL) & 0x70) == 0x30
      && (*(_BYTE *)(*(_QWORD *)(j + 168) + 17LL) & 0x70) == 48;
}
