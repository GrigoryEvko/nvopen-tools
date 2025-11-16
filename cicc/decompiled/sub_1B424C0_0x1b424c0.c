// Function: sub_1B424C0
// Address: 0x1b424c0
//
__int64 __fastcall sub_1B424C0(__int64 *a1, __int64 *a2)
{
  __int64 v2; // rdi
  __int64 v3; // rsi

  v2 = *a1;
  v3 = *a2;
  if ( v2 == v3 )
    return 0;
  else
    return (((int)sub_16A9900(v2 + 24, (unsigned __int64 *)(v3 + 24)) >> 31) & 2u) - 1;
}
