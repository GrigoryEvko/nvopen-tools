// Function: sub_DF9CF0
// Address: 0xdf9cf0
//
__int64 __fastcall sub_DF9CF0(__int64 *a1, unsigned __int8 a2)
{
  __int64 v2; // rdi
  __int64 (*v3)(); // rax

  v2 = *a1;
  v3 = *(__int64 (**)())(*(_QWORD *)v2 + 400LL);
  if ( v3 == sub_DF5DA0 )
    return 2;
  else
    return ((__int64 (__fastcall *)(__int64, _QWORD))v3)(v2, a2);
}
