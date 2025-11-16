// Function: sub_321DF40
// Address: 0x321df40
//
bool __fastcall sub_321DF40(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // r13
  bool result; // al
  _BYTE v4[8]; // [rsp+0h] [rbp-40h] BYREF
  unsigned __int64 v5; // [rsp+8h] [rbp-38h]
  bool v6; // [rsp+10h] [rbp-30h]

  v2 = 0;
  sub_AF47B0(
    (__int64)v4,
    *(unsigned __int64 **)(*(_QWORD *)(a1 + 8) + 16LL),
    *(unsigned __int64 **)(*(_QWORD *)(a1 + 8) + 24LL));
  if ( v6 )
    v2 = v5;
  sub_AF47B0(
    (__int64)v4,
    *(unsigned __int64 **)(*(_QWORD *)(a2 + 8) + 16LL),
    *(unsigned __int64 **)(*(_QWORD *)(a2 + 8) + 24LL));
  result = v6;
  if ( v6 )
    return v5 > v2;
  return result;
}
