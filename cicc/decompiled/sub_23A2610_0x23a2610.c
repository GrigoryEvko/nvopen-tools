// Function: sub_23A2610
// Address: 0x23a2610
//
unsigned __int64 __fastcall sub_23A2610(unsigned __int64 *a1)
{
  _QWORD *v1; // rax
  unsigned __int64 result; // rax
  _QWORD *v3; // [rsp+0h] [rbp-20h] BYREF
  char v4; // [rsp+8h] [rbp-18h]

  v1 = (_QWORD *)sub_22077B0(0x10u);
  if ( v1 )
    *v1 = &unk_4A0EE38;
  v3 = v1;
  v4 = 0;
  result = sub_23571D0(a1, (__int64 *)&v3);
  if ( v3 )
    return (*(__int64 (__fastcall **)(_QWORD *))(*v3 + 8LL))(v3);
  return result;
}
