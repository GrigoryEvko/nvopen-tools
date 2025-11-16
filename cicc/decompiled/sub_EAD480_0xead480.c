// Function: sub_EAD480
// Address: 0xead480
//
__int64 __fastcall sub_EAD480(__int64 a1, __int64 a2)
{
  __int64 v3; // [rsp+18h] [rbp-68h] BYREF
  __int64 v4; // [rsp+20h] [rbp-60h] BYREF
  __int64 v5; // [rsp+28h] [rbp-58h] BYREF
  _QWORD v6[4]; // [rsp+30h] [rbp-50h] BYREF
  char v7; // [rsp+50h] [rbp-30h]
  char v8; // [rsp+51h] [rbp-2Fh]

  v3 = 0;
  v4 = 0;
  v5 = 0;
  if ( (unsigned __int8)sub_EAD290(a1, &v3, a2) )
    return 1;
  v8 = 1;
  v7 = 3;
  v6[0] = "expected comma";
  if ( (unsigned __int8)sub_ECE210(a1, 26, v6) )
    return 1;
  if ( (unsigned __int8)sub_EAC8B0(a1, &v4) )
    return 1;
  v8 = 1;
  v6[0] = "expected comma";
  v7 = 3;
  if ( (unsigned __int8)sub_ECE210(a1, 26, v6)
    || (unsigned __int8)sub_EAC8B0(a1, &v5)
    || (unsigned __int8)sub_ECE000(a1) )
  {
    return 1;
  }
  (*(void (__fastcall **)(_QWORD, __int64, __int64, __int64, __int64))(**(_QWORD **)(a1 + 232) + 888LL))(
    *(_QWORD *)(a1 + 232),
    v3,
    v4,
    v5,
    a2);
  return 0;
}
