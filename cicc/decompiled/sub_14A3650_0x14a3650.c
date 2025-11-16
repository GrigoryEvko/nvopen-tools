// Function: sub_14A3650
// Address: 0x14a3650
//
__int64 __fastcall sub_14A3650(__int64 *a1, __int64 a2, __int64 a3, unsigned __int8 a4)
{
  __int64 v4; // rdi
  __int64 (*v5)(); // rax

  v4 = *a1;
  v5 = *(__int64 (**)())(*(_QWORD *)v4 + 672LL);
  if ( v5 == sub_14A0A70 )
    return 1;
  else
    return ((__int64 (__fastcall *)(__int64, __int64, __int64, _QWORD))v5)(v4, a2, a3, a4);
}
