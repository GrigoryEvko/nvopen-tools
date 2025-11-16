// Function: sub_2C965A0
// Address: 0x2c965a0
//
void __fastcall sub_2C965A0(__int64 a1, _QWORD *a2)
{
  _BYTE *v3; // rsi

  v3 = *(_BYTE **)(a1 + 8);
  if ( v3 == *(_BYTE **)(a1 + 16) )
  {
    sub_2C96410(a1, v3, a2);
  }
  else
  {
    if ( v3 )
    {
      *(_QWORD *)v3 = *a2;
      v3 = *(_BYTE **)(a1 + 8);
    }
    *(_QWORD *)(a1 + 8) = v3 + 8;
  }
}
