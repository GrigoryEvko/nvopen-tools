// Function: sub_1280D20
// Address: 0x1280d20
//
__int64 __fastcall sub_1280D20(__int64 a1, _QWORD *a2, unsigned __int64 *a3)
{
  __int64 v4; // rax

  if ( sub_127B420(*a3) )
  {
    sub_12A6CC0(a1, a2, a3);
  }
  else
  {
    v4 = sub_128F980(a2, a3);
    sub_12800D0(a1, a2, *a3, v4);
  }
  return a1;
}
