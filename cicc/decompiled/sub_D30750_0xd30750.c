// Function: sub_D30750
// Address: 0xd30750
//
bool __fastcall sub_D30750(unsigned __int8 *a1, unsigned __int8 *a2, __int64 a3)
{
  unsigned __int8 *v3; // r13
  _QWORD *v6; // rax
  __int64 v7; // rax

  if ( *a2 == 20 )
    return 1;
  if ( *a2 <= 0x15u )
  {
    v6 = (_QWORD *)sub_BD5C60((__int64)a2);
    v7 = sub_BCB2B0(v6);
    if ( sub_D30730((__int64)a2, v7, a3, 0, 0, 0, 0) )
      return 1;
  }
  v3 = sub_98B9F0(a1);
  return v3 == sub_98B9F0(a2);
}
