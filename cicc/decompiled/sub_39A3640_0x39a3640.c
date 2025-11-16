// Function: sub_39A3640
// Address: 0x39a3640
//
__int64 __fastcall sub_39A3640(__int64 a1, __int64 a2, __int16 a3, __int64 a4)
{
  __int64 *v6; // rsi
  int v8[9]; // [rsp+Ch] [rbp-24h] BYREF

  v6 = (__int64 *)(a2 + 8);
  if ( (unsigned __int16)sub_398C0A0(*(_QWORD *)(a1 + 200)) <= 3u )
    v8[0] = 65542;
  else
    v8[0] = 65559;
  return sub_39A3560(a1, v6, a3, (__int64)v8, a4);
}
