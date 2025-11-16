// Function: sub_14A2C80
// Address: 0x14a2c80
//
__int64 __fastcall sub_14A2C80(
        __int64 *a1,
        __int64 a2,
        unsigned __int64 a3,
        unsigned __int64 a4,
        unsigned __int8 a5,
        unsigned __int64 a6)
{
  __int64 v6; // rdi
  __int64 (__fastcall *v7)(__int64, __int64, unsigned __int64, unsigned __int64, __int64, unsigned __int64); // rax
  __int64 result; // rax

  v6 = *a1;
  v7 = *(__int64 (__fastcall **)(__int64, __int64, unsigned __int64, unsigned __int64, __int64, unsigned __int64))(*(_QWORD *)v6 + 264LL);
  if ( v7 != sub_14A0890 )
    return ((__int64 (__fastcall *)(__int64, __int64, unsigned __int64, unsigned __int64, _QWORD))v7)(
             v6,
             a2,
             a3,
             a4,
             a5);
  result = 0xFFFFFFFFLL;
  if ( __PAIR128__(a4, a3) == 0 )
    return (unsigned int)-(a6 > 1);
  return result;
}
