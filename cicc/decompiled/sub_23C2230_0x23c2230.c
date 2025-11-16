// Function: sub_23C2230
// Address: 0x23c2230
//
__int64 __fastcall sub_23C2230(__int64 a1, const void *a2, size_t a3, __int64 *a4)
{
  __int64 v4; // rax
  __int64 result; // rax
  __int64 v6; // [rsp+8h] [rbp-8h] BYREF

  v4 = *a4;
  *a4 = 0;
  v6 = v4;
  result = sub_23C18D0(a1, a2, a3, &v6);
  if ( v6 )
    return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v6 + 8LL))(v6);
  return result;
}
