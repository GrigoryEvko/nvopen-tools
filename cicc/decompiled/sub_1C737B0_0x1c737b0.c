// Function: sub_1C737B0
// Address: 0x1c737b0
//
bool __fastcall sub_1C737B0(__int64 a1, __int64 a2, __int64 a3, _QWORD *a4)
{
  int v5; // eax
  bool result; // al

  v5 = *(unsigned __int16 *)(a2 + 18);
  if ( (v5 & 0x7FFB) != 0x22 )
  {
    BYTE1(v5) &= ~0x80u;
    if ( ((v5 - 36) & 0xFFFFFFFB) == 0 && a3 == *(_QWORD *)(a2 - 24) )
    {
      result = sub_13FC1A0(a1, *(_QWORD *)(a2 - 48));
      if ( result )
      {
        *a4 = *(_QWORD *)(a2 - 48);
        return result;
      }
    }
    return 0;
  }
  if ( a3 != *(_QWORD *)(a2 - 48) )
    return 0;
  result = sub_13FC1A0(a1, *(_QWORD *)(a2 - 24));
  if ( !result )
    return 0;
  *a4 = *(_QWORD *)(a2 - 24);
  return result;
}
