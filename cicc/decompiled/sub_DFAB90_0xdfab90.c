// Function: sub_DFAB90
// Address: 0xdfab90
//
__int64 __fastcall sub_DFAB90(__int64 *a1, unsigned __int8 a2)
{
  __int64 v2; // rdi
  __int64 (*v3)(); // rax

  v2 = *a1;
  v3 = *(__int64 (**)())(*(_QWORD *)v2 + 832LL);
  if ( v3 == sub_DF5EF0 )
    return 0;
  else
    return ((__int64 (__fastcall *)(__int64, _QWORD))v3)(v2, a2);
}
