// Function: sub_14A2E40
// Address: 0x14a2e40
//
__int64 __fastcall sub_14A2E40(__int64 *a1, __int64 a2, unsigned __int8 a3, unsigned __int8 a4)
{
  __int64 v4; // rdi
  __int64 (*v5)(); // rax

  v4 = *a1;
  v5 = *(__int64 (**)())(*(_QWORD *)v4 + 352LL);
  if ( v5 == sub_14A08E0 )
    return 0;
  else
    return ((__int64 (__fastcall *)(__int64, __int64, _QWORD, _QWORD))v5)(v4, a2, a3, a4);
}
