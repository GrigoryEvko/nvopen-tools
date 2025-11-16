// Function: sub_2579EE0
// Address: 0x2579ee0
//
char __fastcall sub_2579EE0(_BYTE *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 (__fastcall *v6)(__int64); // rax
  char result; // al

  v6 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL);
  if ( v6 == sub_2505E60 )
  {
    result = a1[17];
    if ( !result )
      return result;
    return sub_2579B00((__int64)a1, a2, a3, a4, a5, a6);
  }
  result = ((__int64 (*)(void))v6)();
  if ( result )
    return sub_2579B00((__int64)a1, a2, a3, a4, a5, a6);
  return result;
}
