// Function: sub_1F3CF40
// Address: 0x1f3cf40
//
__int64 __fastcall sub_1F3CF40(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, unsigned int a5, __int64 a6)
{
  __int64 (*v6)(); // rax

  v6 = *(__int64 (**)())(*(_QWORD *)a1 + 880LL);
  if ( v6 == sub_1D5A410 )
    return 0;
  else
    return ((__int64 (__fastcall *)(__int64, _QWORD, __int64, _QWORD, __int64))v6)(a1, a3, a4, a5, a6);
}
