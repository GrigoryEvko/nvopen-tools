// Function: sub_34D0AA0
// Address: 0x34d0aa0
//
__int64 __fastcall sub_34D0AA0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v3; // rsi
  __int64 v4; // rdi

  v3 = (__int64 *)a3;
  v4 = a1 + 8;
  if ( (unsigned int)*(unsigned __int8 *)(a3 + 8) - 17 <= 1 )
    v3 = **(__int64 ***)(a3 + 16);
  return (unsigned int)sub_34D06B0(v4, v3);
}
