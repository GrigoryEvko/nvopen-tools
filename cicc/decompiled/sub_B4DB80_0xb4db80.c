// Function: sub_B4DB80
// Address: 0xb4db80
//
__int64 __fastcall sub_B4DB80(__int64 a1, __int64 a2)
{
  int v2; // eax
  __int64 v4; // rsi
  int v5; // ecx

  v2 = *(unsigned __int8 *)(a1 + 8);
  if ( (_BYTE)v2 == 15 )
  {
    if ( (unsigned __int8)sub_BCBB10() )
      return sub_BCBAE0(a1, a2);
    return 0;
  }
  v4 = *(_QWORD *)(a2 + 8);
  v5 = *(unsigned __int8 *)(v4 + 8);
  if ( (unsigned int)(v5 - 17) <= 1 )
    LOBYTE(v5) = *(_BYTE *)(**(_QWORD **)(v4 + 16) + 8LL);
  if ( (_BYTE)v5 != 12 || (_BYTE)v2 != 16 && (unsigned int)(v2 - 17) > 1 )
    return 0;
  return *(_QWORD *)(a1 + 24);
}
