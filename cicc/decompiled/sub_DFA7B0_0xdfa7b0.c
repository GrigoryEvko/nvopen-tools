// Function: sub_DFA7B0
// Address: 0xdfa7b0
//
__int64 __fastcall sub_DFA7B0(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned __int8 a6,
        __int64 a7)
{
  __int64 v7; // rdi
  __int64 (*v8)(); // r11

  v7 = *a1;
  v8 = *(__int64 (**)())(*(_QWORD *)v7 + 672LL);
  if ( v8 == sub_DF5E70 )
    return 0;
  else
    return ((__int64 (__fastcall *)(__int64, __int64, __int64, __int64, __int64, _QWORD, __int64))v8)(
             v7,
             a2,
             a3,
             a4,
             a5,
             a6,
             a7);
}
