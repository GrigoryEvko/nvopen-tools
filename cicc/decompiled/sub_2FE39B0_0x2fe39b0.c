// Function: sub_2FE39B0
// Address: 0x2fe39b0
//
__int64 __fastcall sub_2FE39B0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  __int64 (*v8)(); // rax

  v8 = *(__int64 (**)())(*(_QWORD *)a1 + 1560LL);
  if ( v8 == sub_2D566A0 )
    return 0;
  else
    return ((__int64 (__fastcall *)(__int64, __int64, __int64, __int64, __int64))v8)(a1, a4, a5, a7, a8);
}
