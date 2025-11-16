// Function: sub_1E70960
// Address: 0x1e70960
//
__int64 __fastcall sub_1E70960(__int64 a1, __int64 a2, char a3)
{
  __int64 result; // rax

  if ( a3 )
    result = sub_1E704A0(a1, a2);
  else
    result = sub_1E70570(a1, a2);
  *(_BYTE *)(a2 + 229) |= 4u;
  return result;
}
