// Function: sub_23A2390
// Address: 0x23a2390
//
unsigned __int64 __fastcall sub_23A2390(unsigned __int64 *a1, __int16 *a2)
{
  __int16 v2; // bx
  __int64 v3; // rax
  unsigned __int64 result; // rax
  unsigned __int64 v5[3]; // [rsp+8h] [rbp-18h] BYREF

  v2 = *a2;
  v3 = sub_22077B0(0x10u);
  if ( v3 )
  {
    *(_WORD *)(v3 + 8) = v2;
    *(_QWORD *)v3 = &unk_4A0DB38;
  }
  v5[0] = v3;
  result = sub_23A2230(a1, v5);
  if ( v5[0] )
    return (*(__int64 (__fastcall **)(unsigned __int64))(*(_QWORD *)v5[0] + 8LL))(v5[0]);
  return result;
}
