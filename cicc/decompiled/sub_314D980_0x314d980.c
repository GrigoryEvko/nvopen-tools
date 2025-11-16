// Function: sub_314D980
// Address: 0x314d980
//
unsigned __int64 __fastcall sub_314D980(unsigned __int64 *a1)
{
  _QWORD *v1; // rax
  unsigned __int64 result; // rax
  _QWORD *v3; // [rsp+8h] [rbp-18h] BYREF

  v1 = (_QWORD *)sub_22077B0(0x10u);
  if ( v1 )
    *v1 = &unk_4A0FFF8;
  v3 = v1;
  result = sub_314D790(a1, (unsigned __int64 *)&v3);
  if ( v3 )
    return (*(__int64 (__fastcall **)(_QWORD *))(*v3 + 8LL))(v3);
  return result;
}
