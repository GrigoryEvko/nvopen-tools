// Function: sub_E31E00
// Address: 0xe31e00
//
void __fastcall sub_E31E00(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // rax
  unsigned __int64 v3; // rbx

  if ( a2 )
  {
    v2 = *(_QWORD *)(a1 + 16);
    if ( a2 - 1 < v2 )
    {
      v3 = v2 - a2;
      sub_E31570(a1, 39);
      if ( v3 > 0x19 )
      {
        sub_E31570(a1, 122);
        sub_E31D10(a1, v3 - 25);
      }
      else
      {
        sub_E31570(a1, v3 + 97);
      }
    }
    else
    {
      *(_BYTE *)(a1 + 49) = 1;
    }
  }
  else
  {
    sub_E31C60(a1, 2u, &unk_3F7CC6C);
  }
}
