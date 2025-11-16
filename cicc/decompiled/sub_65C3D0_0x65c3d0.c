// Function: sub_65C3D0
// Address: 0x65c3d0
//
void __fastcall sub_65C3D0(__int64 a1)
{
  __int64 v2; // rdi
  __int64 v3; // rax
  __int64 v4; // rsi

  if ( (*(_BYTE *)(a1 + 125) & 4) != 0
    || (v4 = *(_QWORD *)(a1 + 312), v2 = *(_QWORD *)(a1 + 304), (unsigned int)sub_65C2A0(v2, v4)) )
  {
    if ( !*(_QWORD *)(a1 + 296) )
      return;
    v2 = *(_QWORD *)a1;
    if ( *(_BYTE *)(*(_QWORD *)a1 + 80LL) == 9 )
    {
      if ( !(unsigned int)sub_64E420(v2, *(_QWORD *)(a1 + 288), (unsigned int *)(a1 + 48)) )
        return;
    }
    else
    {
      v2 = a1;
      if ( (unsigned int)sub_646C60(a1) )
        return;
    }
  }
  *(_WORD *)(a1 + 124) &= 0xF87Fu;
  v3 = sub_72C930(v2);
  *(_QWORD *)(a1 + 288) = v3;
  *(_QWORD *)(a1 + 312) = v3;
  *(_QWORD *)(a1 + 272) = v3;
}
