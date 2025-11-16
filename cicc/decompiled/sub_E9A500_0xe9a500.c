// Function: sub_E9A500
// Address: 0xe9a500
//
void (*__fastcall sub_E9A500(__int64 a1, __int64 a2, unsigned int a3, char a4))()
{
  __int64 v4; // rax
  void (*result)(); // rax
  __int64 (__fastcall *v7)(__int64, unsigned __int8 *); // rbx
  unsigned __int8 *v8; // rsi

  v4 = *(_QWORD *)a1;
  if ( a4 )
  {
    result = *(void (**)())(v4 + 368);
    if ( result != nullsub_351 )
      return (void (*)())((__int64 (__fastcall *)(__int64, __int64, _QWORD))result)(a1, a2, 0);
  }
  else
  {
    v7 = *(__int64 (__fastcall **)(__int64, unsigned __int8 *))(v4 + 528);
    v8 = (unsigned __int8 *)sub_E808D0(a2, 0, *(_QWORD **)(a1 + 8), 0);
    if ( v7 == sub_E9A480 )
      return (void (*)())sub_E9A370(a1, v8);
    else
      return (void (*)())((__int64 (__fastcall *)(__int64, unsigned __int8 *, _QWORD, _QWORD))v7)(a1, v8, a3, 0);
  }
  return result;
}
