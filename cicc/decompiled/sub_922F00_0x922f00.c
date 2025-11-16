// Function: sub_922F00
// Address: 0x922f00
//
__int64 __fastcall sub_922F00(__int64 a1, __int64 a2, unsigned __int64 *a3)
{
  __int64 v4; // rax

  if ( sub_91B770(*a3) )
  {
    sub_947F00(a1, a2, a3);
  }
  else
  {
    v4 = sub_92F410(a2, a3);
    sub_922010(a1, a2, *a3, v4);
  }
  return a1;
}
