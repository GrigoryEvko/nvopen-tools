// Function: sub_6E72F0
// Address: 0x6e72f0
//
__int64 __fastcall sub_6E72F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rax
  __int64 result; // rax

  sub_6E2E50(6, a4);
  v6 = sub_72CBA0();
  *(_QWORD *)(a4 + 136) = a1;
  *(_QWORD *)a4 = v6;
  *(_QWORD *)(a4 + 88) = a2;
  result = *(_QWORD *)(a3 + 8);
  *(_QWORD *)(a4 + 112) = result;
  if ( (*(_BYTE *)(a3 + 18) & 1) != 0 )
  {
    *(_BYTE *)(a4 + 19) |= 8u;
    result = *(_QWORD *)(a3 + 40);
    *(_QWORD *)(a4 + 104) = result;
  }
  return result;
}
