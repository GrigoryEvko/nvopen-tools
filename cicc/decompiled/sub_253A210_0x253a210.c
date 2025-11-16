// Function: sub_253A210
// Address: 0x253a210
//
__int64 __fastcall sub_253A210(__int64 a1, __int64 a2)
{
  char v2; // r8
  __int64 result; // rax
  char v4; // [rsp+Bh] [rbp-2Dh] BYREF
  __int64 v5; // [rsp+Ch] [rbp-2Ch] BYREF
  int v6; // [rsp+14h] [rbp-24h]
  _QWORD v7[3]; // [rsp+18h] [rbp-20h] BYREF

  v5 = 0xB00000005LL;
  v7[0] = a2;
  v7[1] = a1;
  v4 = 0;
  v6 = 56;
  v2 = sub_2526370(
         a2,
         (__int64 (__fastcall *)(__int64, unsigned __int64, __int64))sub_257B3C0,
         (__int64)v7,
         a1,
         (int *)&v5,
         3,
         &v4,
         0,
         0);
  result = 1;
  if ( !v2 )
  {
    *(_BYTE *)(a1 + 97) = *(_BYTE *)(a1 + 96);
    return 0;
  }
  return result;
}
