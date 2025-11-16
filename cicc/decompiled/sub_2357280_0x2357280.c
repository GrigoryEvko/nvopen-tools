// Function: sub_2357280
// Address: 0x2357280
//
unsigned __int64 __fastcall sub_2357280(unsigned __int64 *a1, __int64 *a2)
{
  __int64 v2; // r13
  _QWORD *v3; // rax
  unsigned __int64 result; // rax
  unsigned __int64 v5[3]; // [rsp+8h] [rbp-18h] BYREF

  v2 = *a2;
  *a2 = 0;
  v3 = (_QWORD *)sub_22077B0(0x10u);
  if ( v3 )
  {
    v3[1] = v2;
    v5[0] = (unsigned __int64)v3;
    *v3 = &unk_4A0C3F8;
    result = sub_2356EF0(a1, v5);
    if ( v5[0] )
      return (*(__int64 (__fastcall **)(unsigned __int64))(*(_QWORD *)v5[0] + 8LL))(v5[0]);
  }
  else
  {
    v5[0] = 0;
    result = sub_2356EF0(a1, v5);
    if ( v5[0] )
      result = (*(__int64 (__fastcall **)(unsigned __int64))(*(_QWORD *)v5[0] + 8LL))(v5[0]);
    if ( v2 )
      return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v2 + 8LL))(v2);
  }
  return result;
}
