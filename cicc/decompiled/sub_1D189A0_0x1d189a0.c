// Function: sub_1D189A0
// Address: 0x1d189a0
//
unsigned __int64 __fastcall sub_1D189A0(__int64 a1)
{
  unsigned __int64 result; // rax
  __int64 v2; // rsi

  result = sub_1D18290(a1);
  v2 = *(_QWORD *)(a1 + 72);
  if ( v2 )
    return sub_161E7C0(a1 + 72, v2);
  return result;
}
