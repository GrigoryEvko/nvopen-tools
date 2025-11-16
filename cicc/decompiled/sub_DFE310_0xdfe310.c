// Function: sub_DFE310
// Address: 0xdfe310
//
__int64 __fastcall sub_DFE310(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 1552LL);
  if ( v1 == sub_DF5CE0 )
    return 0;
  else
    return v1();
}
