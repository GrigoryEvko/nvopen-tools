// Function: sub_11D9780
// Address: 0x11d9780
//
__int64 **__fastcall sub_11D9780(__int64 *a1, unsigned __int8 *a2, __int64 a3)
{
  __int64 v4; // r14
  unsigned __int8 *v5; // rax

  v4 = sub_B140C0(a3);
  if ( !(unsigned __int8)sub_11D3030(a1, v4) )
    return (__int64 **)sub_B13710(a3);
  v5 = (unsigned __int8 *)sub_11D6F60(a1, v4);
  return sub_B13360(a3, a2, v5, 0);
}
