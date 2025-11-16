// Function: sub_E7F4D0
// Address: 0xe7f4d0
//
__int64 __fastcall sub_E7F4D0(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  sub_E8DC70();
  result = *(_QWORD *)(*(_QWORD *)(a1 + 288) + 8LL);
  if ( (*(_BYTE *)(result + 153) & 4) != 0 )
    return sub_EA15B0(a2, 6);
  return result;
}
