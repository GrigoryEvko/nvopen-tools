// Function: sub_7AC9D0
// Address: 0x7ac9d0
//
__int64 __fastcall sub_7AC9D0(__int64 a1)
{
  char v1; // al
  unsigned int v2; // r8d
  __int64 v4; // rax
  __int64 v5; // rdx

  v1 = *(_BYTE *)(a1 + 80);
  v2 = 1;
  if ( v1 == 19 )
    return v2;
  if ( v1 == 3 )
  {
    v2 = 0;
    if ( *(_BYTE *)(a1 + 104) )
    {
      v4 = *(_QWORD *)(a1 + 88);
      if ( (*(_BYTE *)(v4 + 177) & 0x10) != 0 )
        return *(_QWORD *)(*(_QWORD *)(v4 + 168) + 168LL) != 0;
    }
    return v2;
  }
  if ( (unsigned __int8)(v1 - 20) <= 1u )
    return v2;
  if ( ((v1 - 7) & 0xFD) == 0 )
  {
    v5 = *(_QWORD *)(a1 + 88);
    v2 = 0;
    if ( !v5 )
      return v2;
    if ( (*(_BYTE *)(v5 + 170) & 0x10) != 0 )
    {
      v2 = 1;
      if ( **(_QWORD **)(v5 + 216) )
        return v2;
    }
  }
  v2 = 0;
  if ( v1 != 17 )
    return v2;
  return (unsigned int)sub_8780F0(a1) != 0;
}
