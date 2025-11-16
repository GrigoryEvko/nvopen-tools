// Function: sub_14A2A90
// Address: 0x14a2a90
//
char __fastcall sub_14A2A90(
        __int64 *a1,
        __int64 a2,
        unsigned __int64 a3,
        unsigned __int64 a4,
        unsigned __int8 a5,
        unsigned __int64 a6)
{
  __int64 v6; // rdi
  bool (__fastcall *v7)(__int64, __int64, unsigned __int64, unsigned __int64, __int64, unsigned __int64); // rax
  char result; // al

  v6 = *a1;
  v7 = *(bool (__fastcall **)(__int64, __int64, unsigned __int64, unsigned __int64, __int64, unsigned __int64))(*(_QWORD *)v6 + 176LL);
  if ( v7 != sub_14A0850 )
    return ((__int64 (__fastcall *)(__int64, __int64, unsigned __int64, unsigned __int64, _QWORD))v7)(
             v6,
             a2,
             a3,
             a4,
             a5);
  result = 0;
  if ( __PAIR128__(a4, a3) == 0 )
    return a6 <= 1;
  return result;
}
