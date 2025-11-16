// Function: sub_2535970
// Address: 0x2535970
//
__int64 __fastcall sub_2535970(__int64 a1, __int64 a2)
{
  char v2; // r8
  __int64 result; // rax
  _QWORD v4[3]; // [rsp+0h] [rbp-20h] BYREF

  v4[0] = a2;
  v4[1] = a1;
  v2 = sub_2527330(a2, (unsigned __int8 (__fastcall *)(__int64, _QWORD))sub_2597270, (__int64)v4, a1, 1u, 1u);
  result = 1;
  if ( !v2 )
  {
    *(_BYTE *)(a1 + 97) = *(_BYTE *)(a1 + 96);
    return 0;
  }
  return result;
}
