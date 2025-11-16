// Function: sub_10569D0
// Address: 0x10569d0
//
__int64 __fastcall sub_10569D0(__int64 a1)
{
  __int64 v1; // rsi
  __int64 result; // rax
  __int64 v3; // [rsp+8h] [rbp-18h] BYREF

  v1 = *(_QWORD *)(a1 + 184);
  v3 = 0;
  *(_QWORD *)(a1 + 184) = 0;
  if ( v1 )
  {
    result = sub_10568E0(a1 + 184, v1);
    if ( v3 )
      result = sub_10568E0((__int64)&v3, v3);
  }
  *(_QWORD *)(a1 + 176) = 0;
  return result;
}
