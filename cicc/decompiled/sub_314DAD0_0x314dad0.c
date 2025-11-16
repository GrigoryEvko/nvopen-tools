// Function: sub_314DAD0
// Address: 0x314dad0
//
unsigned __int64 __fastcall sub_314DAD0(unsigned __int64 *a1)
{
  _QWORD *v1; // rax
  unsigned __int64 result; // rax
  _QWORD *v3; // [rsp+8h] [rbp-18h] BYREF

  v1 = (_QWORD *)sub_22077B0(0x10u);
  if ( v1 )
    *v1 = &unk_4A30EE0;
  v3 = v1;
  result = sub_314D9D0(a1, (unsigned __int64 *)&v3);
  if ( v3 )
    return (*(__int64 (__fastcall **)(_QWORD *))(*v3 + 8LL))(v3);
  return result;
}
