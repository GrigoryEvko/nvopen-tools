// Function: sub_DF9C90
// Address: 0xdf9c90
//
__int64 __fastcall sub_DF9C90(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 384LL);
  if ( v1 == sub_DF5D90 )
    return 16;
  else
    return v1();
}
