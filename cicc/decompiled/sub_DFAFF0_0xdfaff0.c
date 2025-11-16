// Function: sub_DFAFF0
// Address: 0xdfaff0
//
__int64 __fastcall sub_DFAFF0(__int64 a1)
{
  __int64 (*v1)(void); // r8

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 944LL);
  if ( v1 == sub_DF5F90 )
    return 1;
  else
    return v1();
}
