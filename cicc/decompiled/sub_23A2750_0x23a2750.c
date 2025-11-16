// Function: sub_23A2750
// Address: 0x23a2750
//
unsigned __int64 __fastcall sub_23A2750(__int64 a1, unsigned __int64 *a2)
{
  _QWORD *v2; // rax
  _QWORD *v3; // rax
  unsigned __int64 result; // rax
  unsigned __int64 v5[3]; // [rsp+8h] [rbp-18h] BYREF

  v2 = (_QWORD *)sub_22077B0(0x10u);
  if ( v2 )
    *v2 = &unk_4A0CF38;
  v5[0] = (unsigned __int64)v2;
  sub_23A2230(a2, v5);
  if ( v5[0] )
    (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v5[0] + 8LL))(v5[0]);
  v3 = (_QWORD *)sub_22077B0(0x10u);
  if ( v3 )
    *v3 = &unk_4A0D9B8;
  v5[0] = (unsigned __int64)v3;
  result = sub_23A2230(a2, v5);
  if ( v5[0] )
    return (*(__int64 (__fastcall **)(unsigned __int64))(*(_QWORD *)v5[0] + 8LL))(v5[0]);
  return result;
}
