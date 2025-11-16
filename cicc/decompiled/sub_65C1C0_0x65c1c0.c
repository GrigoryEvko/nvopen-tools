// Function: sub_65C1C0
// Address: 0x65c1c0
//
__int64 __fastcall sub_65C1C0(__int64 a1)
{
  __int64 result; // rax
  char v2; // dl

  result = *(_QWORD *)(a1 + 352);
  if ( result )
  {
    v2 = *(_BYTE *)(result + 16);
    if ( v2 == 53 )
    {
      result = *(_QWORD *)(result + 24);
      *(_BYTE *)(result + 57) |= 0x80u;
    }
    else if ( v2 )
    {
      result = *(_QWORD *)(result + 24);
      *(_BYTE *)(result + 90) |= 1u;
    }
    else if ( *(char *)(a1 + 125) < 0 )
    {
      result = *(_QWORD *)(*(_QWORD *)a1 + 88LL);
      *(_BYTE *)(result + 42) |= 0x10u;
    }
  }
  return result;
}
