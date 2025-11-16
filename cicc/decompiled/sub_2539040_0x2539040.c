// Function: sub_2539040
// Address: 0x2539040
//
__int64 __fastcall sub_2539040(__int64 a1, __int64 a2)
{
  char v2; // r8
  __int64 result; // rax
  char v4; // [rsp+12h] [rbp-16h] BYREF
  char v5; // [rsp+13h] [rbp-15h] BYREF
  int v6[3]; // [rsp+14h] [rbp-14h] BYREF

  v5 = 0;
  v6[0] = 1;
  v2 = sub_2526370(
         a2,
         (__int64 (__fastcall *)(__int64, unsigned __int64, __int64))sub_2535060,
         (__int64)&v4,
         a1,
         v6,
         1,
         &v5,
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
