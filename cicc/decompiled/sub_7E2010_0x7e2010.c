// Function: sub_7E2010
// Address: 0x7e2010
//
void __fastcall sub_7E2010(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int8 v6; // al

  if ( *(_BYTE *)(a1 + 24) == 1 )
  {
    sub_7DFF80(a1, a2, a3, a4, a5, a6);
    if ( *(_BYTE *)(a1 + 24) == 1 )
    {
      v6 = *(_BYTE *)(a1 + 57);
      if ( v6 == 13 )
      {
        *(_BYTE *)(a1 + 57) = (unsigned int)sub_7E1F90(**(_QWORD **)(a1 + 72)) == 0 ? 2 : 10;
      }
      else if ( v6 > 0xDu )
      {
        if ( v6 == 19 )
          *(_BYTE *)(a1 + 57) = 6;
      }
      else if ( v6 == 4 )
      {
        *(_BYTE *)(a1 + 57) = 3;
      }
      else if ( v6 == 5 )
      {
        *(_BYTE *)(a1 + 57) = 10;
      }
    }
  }
}
