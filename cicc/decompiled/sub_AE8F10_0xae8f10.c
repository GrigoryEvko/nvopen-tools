// Function: sub_AE8F10
// Address: 0xae8f10
//
__int64 __fastcall sub_AE8F10(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 result; // rax
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // rsi
  _QWORD v9[5]; // [rsp+8h] [rbp-28h] BYREF

  v3 = sub_B026B0(a2, a3);
  result = sub_B10CB0(v9, v3);
  if ( *(_QWORD *)(a1 + 48) )
    result = sub_B91220(a1 + 48);
  v8 = v9[0];
  *(_QWORD *)(a1 + 48) = v9[0];
  if ( v8 )
    return sub_B976B0(v9, v8, a1 + 48, v5, v6, v7);
  return result;
}
