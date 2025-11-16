// Function: sub_130B460
// Address: 0x130b460
//
__int64 __fastcall sub_130B460(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  *(_BYTE *)(a2 + 16) = 0;
  if ( *(_BYTE *)(a2 + 17) )
  {
    sub_130ECD0(a1, a2 + 62264);
    return sub_1348790(a1, a2 + 62384);
  }
  return result;
}
