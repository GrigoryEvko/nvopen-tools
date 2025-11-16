// Function: sub_130B4C0
// Address: 0x130b4c0
//
__int64 __fastcall sub_130B4C0(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  result = sub_130C8F0(a1, a2 + 24);
  if ( *(_BYTE *)(a2 + 17) )
  {
    sub_130EC10(a1, a2 + 62264);
    return sub_1348790(a1, a2 + 62384);
  }
  return result;
}
