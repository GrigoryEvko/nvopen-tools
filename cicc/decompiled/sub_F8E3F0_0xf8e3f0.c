// Function: sub_F8E3F0
// Address: 0xf8e3f0
//
__int64 __fastcall sub_F8E3F0(__int64 *a1, __int64 *a2)
{
  __int64 v2; // rdi
  __int64 v3; // rsi

  v2 = *a1;
  v3 = *a2;
  if ( v2 == v3 )
    return 0;
  else
    return (((int)sub_C49970(v2 + 24, (unsigned __int64 *)(v3 + 24)) >> 31) & 2u) - 1;
}
