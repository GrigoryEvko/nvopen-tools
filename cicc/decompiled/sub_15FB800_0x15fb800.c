// Function: sub_15FB800
// Address: 0x15fb800
//
__int64 __fastcall sub_15FB800(__int64 a1)
{
  unsigned int v1; // ecx

  v1 = *(unsigned __int8 *)(a1 + 16) - 24;
  if ( v1 > 0x1C )
    return 1;
  if ( ((1LL << v1) & 0x1C019800) == 0 )
    return 1;
  sub_16484A0(a1 - 48, a1 - 24);
  return 0;
}
