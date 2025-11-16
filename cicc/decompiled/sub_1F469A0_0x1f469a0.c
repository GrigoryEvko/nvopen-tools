// Function: sub_1F469A0
// Address: 0x1f469a0
//
__int64 __fastcall sub_1F469A0(__int64 a1, _QWORD *a2)
{
  _QWORD *v2; // rax

  if ( (unsigned int)sub_1F45DD0(a1) && !byte_4FCCFC0 )
  {
    a2 = (_QWORD *)sub_1D67920();
    sub_1F46490(a1, a2, 1, 1, 0);
  }
  v2 = (_QWORD *)sub_1B6D360(a1, (__int64)a2);
  return sub_1F46490(a1, v2, 1, 1, 0);
}
