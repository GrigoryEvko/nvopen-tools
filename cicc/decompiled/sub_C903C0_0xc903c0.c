// Function: sub_C903C0
// Address: 0xc903c0
//
__int64 __fastcall sub_C903C0(__int64 *a1, __int64 a2)
{
  unsigned __int64 v2; // rax

  v2 = *(_QWORD *)(*a1 + 16) - *(_QWORD *)(*a1 + 8);
  if ( v2 <= 0xFF )
    return sub_C8FD40(a1, a2);
  if ( v2 <= 0xFFFF )
    return sub_C90030(a1, a2);
  if ( v2 > 0xFFFFFFFF )
    return sub_C90310(a1, a2);
  return sub_C901A0(a1, a2);
}
