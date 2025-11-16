// Function: sub_EAE0B0
// Address: 0xeae0b0
//
__int64 __fastcall sub_EAE0B0(__int64 a1)
{
  __int64 v1; // r12
  __int64 v3; // [rsp+10h] [rbp-30h] BYREF
  _QWORD v4[5]; // [rsp+18h] [rbp-28h] BYREF

  v1 = sub_ECD690(a1 + 40);
  if ( !*(_BYTE *)(a1 + 869) && (unsigned __int8)sub_EA2540(a1) )
    return 1;
  v4[0] = 0;
  if ( sub_EAC4D0(a1, &v3, (__int64)v4) )
    return 1;
  v4[0] = 0;
  if ( (unsigned __int8)sub_ECE2A0(a1, 26) )
  {
    if ( (unsigned __int8)sub_EAC8B0(a1, v4) )
      return 1;
  }
  if ( (unsigned __int8)sub_ECE000(a1) )
    return 1;
  (*(void (__fastcall **)(_QWORD, __int64, _QWORD, __int64))(**(_QWORD **)(a1 + 232) + 584LL))(
    *(_QWORD *)(a1 + 232),
    v3,
    v4[0],
    v1);
  return 0;
}
