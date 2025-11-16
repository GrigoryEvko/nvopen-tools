// Function: sub_DFAFA0
// Address: 0xdfafa0
//
__int64 __fastcall sub_DFAFA0(__int64 a1)
{
  __int64 (*v1)(void); // r9

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 936LL);
  if ( v1 == sub_DF5F80 )
    return 0;
  else
    return v1();
}
