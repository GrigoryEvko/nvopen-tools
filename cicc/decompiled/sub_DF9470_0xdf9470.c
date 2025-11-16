// Function: sub_DF9470
// Address: 0xdf9470
//
__int64 __fastcall sub_DF9470(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 72LL);
  if ( v1 == sub_DF5BB0 )
    return 0;
  else
    return v1();
}
