// Function: sub_AB07B0
// Address: 0xab07b0
//
__int64 __fastcall sub_AB07B0(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  if ( sub_AAF7D0(a1) || sub_AAF7D0(a2) || sub_AB0760(a1) && sub_AB0760(a2) )
    return 1;
  result = sub_AB06D0(a1);
  if ( (_BYTE)result )
    return sub_AB06D0(a2);
  return result;
}
