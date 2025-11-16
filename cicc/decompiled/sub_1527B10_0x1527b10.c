// Function: sub_1527B10
// Address: 0x1527b10
//
void __fastcall sub_1527B10(_DWORD *a1, __int64 a2, unsigned int a3)
{
  char v3; // al
  unsigned int v4; // esi

  v3 = (*(_BYTE *)(a2 + 8) >> 1) & 7;
  if ( v3 == 2 )
  {
    if ( *(_QWORD *)a2 )
      sub_1525280(a1, a3, *(_QWORD *)a2);
  }
  else if ( v3 == 4 )
  {
    if ( (unsigned __int8)(a3 - 97) <= 0x19u )
    {
      sub_1524D80(a1, (char)a3 - 97, 6);
    }
    else
    {
      if ( (unsigned __int8)(a3 - 65) <= 0x19u )
      {
        v4 = (char)a3 - 39;
      }
      else
      {
        v4 = (char)a3 + 4;
        if ( (unsigned __int8)(a3 - 48) > 9u )
          v4 = ((_BYTE)a3 != 46) + 62;
      }
      sub_1524D80(a1, v4, 6);
    }
  }
  else if ( *(_QWORD *)a2 )
  {
    sub_1524D80(a1, a3, *(_QWORD *)a2);
  }
}
