// Function: sub_DFE370
// Address: 0xdfe370
//
__int64 __fastcall sub_DFE370(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 1576LL);
  if ( v1 == sub_DF5C80 )
    return 1;
  else
    return v1();
}
