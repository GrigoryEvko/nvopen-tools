// Function: sub_E99540
// Address: 0xe99540
//
void (__fastcall *__fastcall sub_E99540(_DWORD *a1))(__int64 a1, __int64 a2)
{
  void (__fastcall *result)(__int64, __int64); // rax
  void (__fastcall *v2)(__int64, __int64); // rsi

  result = (void (__fastcall *)(__int64, __int64))sub_E99320((__int64)a1);
  if ( result )
  {
    v2 = result;
    result = *(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 16LL);
    if ( result == sub_E97640 )
      *((_QWORD *)v2 + 1) = 1;
    else
      result = (void (__fastcall *)(__int64, __int64))((__int64 (__fastcall *)(_DWORD *, _QWORD))result)(a1, v2);
    --a1[14];
  }
  return result;
}
