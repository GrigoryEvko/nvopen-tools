// Function: sub_14A3140
// Address: 0x14a3140
//
__int64 __fastcall sub_14A3140(__int64 *a1, unsigned __int8 a2)
{
  __int64 v2; // rdi
  __int64 (*v3)(); // rax

  v2 = *a1;
  v3 = *(__int64 (**)())(*(_QWORD *)v2 + 480LL);
  if ( v3 == sub_14A0960 )
    return 8;
  else
    return ((__int64 (__fastcall *)(__int64, _QWORD))v3)(v2, a2);
}
