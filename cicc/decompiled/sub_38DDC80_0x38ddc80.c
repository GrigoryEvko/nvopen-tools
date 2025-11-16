// Function: sub_38DDC80
// Address: 0x38ddc80
//
void (*__fastcall sub_38DDC80(__int64 *a1, __int64 a2, unsigned int a3, char a4))()
{
  __int64 v4; // rax
  void (*result)(); // rax
  __int64 (__fastcall *v7)(__int64, unsigned int *); // rbx
  unsigned int *v8; // rsi

  v4 = *a1;
  if ( a4 )
  {
    result = *(void (**)())(v4 + 328);
    if ( result != nullsub_1946 )
      return (void (*)())((__int64 (__fastcall *)(__int64 *, __int64, _QWORD))result)(a1, a2, 0);
  }
  else
  {
    v7 = *(__int64 (__fastcall **)(__int64, unsigned int *))(v4 + 416);
    v8 = (unsigned int *)sub_38CF310(a2, 0, a1[1], 0);
    if ( v7 == sub_38DDC00 )
      return (void (*)())sub_38DDAF0((__int64)a1, v8);
    else
      return (void (*)())((__int64 (__fastcall *)(__int64 *, unsigned int *, _QWORD, _QWORD))v7)(a1, v8, a3, 0);
  }
  return result;
}
