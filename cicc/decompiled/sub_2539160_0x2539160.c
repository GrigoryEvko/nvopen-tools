// Function: sub_2539160
// Address: 0x2539160
//
__int64 __fastcall sub_2539160(__int64 a1, __int64 a2)
{
  char v2; // r8
  __int64 result; // rax
  char v4; // [rsp+17h] [rbp-21h] BYREF
  _QWORD v5[3]; // [rsp+18h] [rbp-20h] BYREF

  v5[0] = a2;
  v5[1] = a1;
  v4 = 0;
  v2 = sub_2526370(
         a2,
         (__int64 (__fastcall *)(__int64, unsigned __int64, __int64))sub_259E190,
         (__int64)v5,
         a1,
         dword_438A640,
         6,
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
