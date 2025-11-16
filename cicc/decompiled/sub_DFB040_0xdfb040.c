// Function: sub_DFB040
// Address: 0xdfb040
//
__int64 __fastcall sub_DFB040(__int64 *a1)
{
  __int64 v1; // rdi
  __int64 (*v2)(); // r10

  v1 = *a1;
  v2 = *(__int64 (**)())(*(_QWORD *)v1 + 952LL);
  if ( v2 == sub_DF5FA0 )
    return 0;
  else
    return ((__int64 (__fastcall *)(__int64))v2)(v1);
}
