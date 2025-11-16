// Function: sub_DFAE30
// Address: 0xdfae30
//
__int64 __fastcall sub_DFAE30(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 872LL);
  if ( v1 == sub_DF5CE0 )
    return 0;
  else
    return v1();
}
