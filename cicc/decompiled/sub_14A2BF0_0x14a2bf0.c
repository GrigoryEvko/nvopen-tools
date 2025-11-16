// Function: sub_14A2BF0
// Address: 0x14a2bf0
//
__int64 __fastcall sub_14A2BF0(__int64 *a1, __int64 a2, unsigned __int8 a3)
{
  __int64 v3; // rdi
  __int64 (*v4)(); // rax

  v3 = *a1;
  v4 = *(__int64 (**)())(*(_QWORD *)v3 + 240LL);
  if ( v4 == sub_14A0860 )
    return 0;
  else
    return ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v4)(v3, a2, a3);
}
