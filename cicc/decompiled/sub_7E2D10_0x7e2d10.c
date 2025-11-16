// Function: sub_7E2D10
// Address: 0x7e2d10
//
__int64 __fastcall sub_7E2D10(__int64 a1)
{
  __int64 result; // rax
  __int64 v3; // rdi

  if ( qword_4D03F68[1] != qword_4F04C50 )
    return sub_7E18B0();
  if ( *(_QWORD *)(a1 + 16) )
    return sub_7E18B0();
  v3 = *(_QWORD *)(qword_4F04C50 + 80LL);
  if ( *(_BYTE *)(v3 + 40) != 11 )
    return sub_7E18B0();
  result = sub_7E2C20(v3);
  if ( a1 != result )
    return sub_7E18B0();
  return result;
}
