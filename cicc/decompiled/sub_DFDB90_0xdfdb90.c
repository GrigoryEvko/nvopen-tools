// Function: sub_DFDB90
// Address: 0xdfdb90
//
__int64 __fastcall sub_DFDB90(__int64 a1)
{
  __int64 (*v1)(void); // r8

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 1384LL);
  if ( v1 == sub_DF6260 )
    return 0;
  else
    return v1();
}
