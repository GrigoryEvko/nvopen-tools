// Function: sub_DFAE90
// Address: 0xdfae90
//
__int64 __fastcall sub_DFAE90(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 888LL);
  if ( v1 == sub_DF5F50 )
    return 0;
  else
    return v1();
}
