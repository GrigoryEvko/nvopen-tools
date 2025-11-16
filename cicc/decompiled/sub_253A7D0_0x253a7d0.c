// Function: sub_253A7D0
// Address: 0x253a7d0
//
__int64 __fastcall sub_253A7D0(__int64 a1, __int64 a2)
{
  char v2; // r8
  __int64 result; // rax
  char v4; // [rsp+3h] [rbp-3Dh] BYREF
  __int64 v5; // [rsp+4h] [rbp-3Ch] BYREF
  int v6; // [rsp+Ch] [rbp-34h]
  _QWORD v7[6]; // [rsp+10h] [rbp-30h] BYREF

  LODWORD(v7[0]) = 19;
  if ( (unsigned __int8)sub_2516400(a2, (__m128i *)(a1 + 72), (__int64)v7, 1, 0, 0)
    && (unsigned __int8)sub_252A800(a2, (__m128i *)(a1 + 72), a1, (bool *)v7) )
  {
    return 1;
  }
  v5 = 0xB00000005LL;
  v7[0] = a2;
  v7[1] = a1;
  v4 = 0;
  v6 = 56;
  v2 = sub_2526370(
         a2,
         (__int64 (__fastcall *)(__int64, unsigned __int64, __int64))sub_259B770,
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
