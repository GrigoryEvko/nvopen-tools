// Function: sub_C50260
// Address: 0xc50260
//
__int64 (__fastcall *__fastcall sub_C50260(__int64 a1, __int64 a2))(__int64, __int64, __int64)
{
  __int64 (__fastcall *result)(__int64, __int64, __int64); // rax
  __int64 v4; // rdi

  *(_QWORD *)a1 = off_49DC4A0;
  result = *(__int64 (__fastcall **)(__int64, __int64, __int64))(a1 + 176);
  if ( result )
  {
    a2 = a1 + 160;
    result = (__int64 (__fastcall *)(__int64, __int64, __int64))result(a2, a2, 3);
  }
  if ( !*(_BYTE *)(a1 + 124) )
    result = (__int64 (__fastcall *)(__int64, __int64, __int64))_libc_free(*(_QWORD *)(a1 + 104), a2);
  v4 = *(_QWORD *)(a1 + 72);
  if ( v4 != a1 + 88 )
    return (__int64 (__fastcall *)(__int64, __int64, __int64))_libc_free(v4, a2);
  return result;
}
