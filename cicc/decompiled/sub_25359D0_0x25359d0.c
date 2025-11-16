// Function: sub_25359D0
// Address: 0x25359d0
//
__int64 __fastcall sub_25359D0(__int64 a1, __int64 a2)
{
  _DWORD v3[11]; // [rsp-2Ch] [rbp-2Ch] BYREF

  if ( !*(_BYTE *)(a1 + 96) )
    return 1;
  v3[0] = 6;
  if ( !(unsigned __int8)sub_2516400(a2, (__m128i *)(a1 + 72), (__int64)v3, 1, 0, 0) )
    return 1;
  v3[0] = 6;
  sub_2515E10(a2, (__int64 *)(a1 + 72), (__int64)v3, 1);
  return 0;
}
