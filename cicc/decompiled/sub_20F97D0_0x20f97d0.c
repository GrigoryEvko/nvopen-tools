// Function: sub_20F97D0
// Address: 0x20f97d0
//
__int64 __fastcall sub_20F97D0(_QWORD *a1)
{
  __int64 v2; // rdi
  __int64 result; // rax
  __int64 v4; // rdi
  __int64 v5; // rdi

  v2 = a1[29];
  a1[29] = 0;
  if ( v2 )
    result = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v2 + 8LL))(v2);
  v4 = a1[30];
  a1[30] = 0;
  if ( v4 )
    result = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v4 + 8LL))(v4);
  v5 = a1[31];
  a1[31] = 0;
  if ( v5 )
    return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v5 + 8LL))(v5);
  return result;
}
