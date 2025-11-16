// Function: sub_22EC8B0
// Address: 0x22ec8b0
//
__int64 __fastcall sub_22EC8B0(__int64 *a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 v5; // r12
  __int64 v6; // rbx
  __int64 result; // rax
  __int64 v8; // rdi
  __int64 v9[7]; // [rsp+8h] [rbp-38h] BYREF

  v5 = *a4;
  *a4 = 0;
  v6 = *a1;
  if ( v5 )
  {
    (*(void (__fastcall **)(__int64 *, __int64))(*(_QWORD *)v5 + 16LL))(v9, v5);
    sub_22EC7B0(v6, a2, a3, v9);
    if ( v9[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v9[0] + 8LL))(v9[0]);
    return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v5 + 8LL))(v5);
  }
  else
  {
    v8 = *a1;
    v9[0] = 0;
    result = sub_22EC7B0(v8, a2, a3, v9);
    if ( v9[0] )
      return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v9[0] + 8LL))(v9[0]);
  }
  return result;
}
