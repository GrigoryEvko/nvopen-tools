// Function: sub_17290A0
// Address: 0x17290a0
//
__int64 __fastcall sub_17290A0(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rsi
  unsigned __int8 v6; // al
  unsigned int v7; // r8d
  unsigned int v8; // r8d

  v5 = *(a2 - 3);
  if ( *(_QWORD *)v5 == *a2 )
  {
    return 0;
  }
  else
  {
    v6 = *(_BYTE *)(v5 + 16);
    v7 = 0;
    if ( v6 > 0x10u )
    {
      v7 = 1;
      if ( (unsigned __int8)(v6 - 60) <= 0xCu )
      {
        LOBYTE(v8) = (unsigned int)sub_174B310(a1, v5, a2, a4, 1) == 0;
        return v8;
      }
    }
  }
  return v7;
}
