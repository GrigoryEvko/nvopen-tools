// Function: sub_17F2260
// Address: 0x17f2260
//
__int64 __fastcall sub_17F2260(__int64 a1, __int64 a2, __int64 a3)
{
  bool v3; // zf
  __int64 result; // rax
  __int64 v5; // [rsp+8h] [rbp-18h] BYREF

  v3 = *(_QWORD *)(a1 + 56) == 0;
  v5 = *(_QWORD *)(a1 + 32);
  if ( v3 )
    sub_4263D6(a1, a2, a3);
  (*(void (__fastcall **)(__int64, __int64 *))(a1 + 64))(a1 + 40, &v5);
  result = *(_QWORD *)(a1 + 24);
  if ( result )
  {
    if ( result != -8 && result != -16 )
      result = sub_1649B30((_QWORD *)(a1 + 8));
    *(_QWORD *)(a1 + 24) = 0;
  }
  return result;
}
