// Function: sub_DFE490
// Address: 0xdfe490
//
__int64 __fastcall sub_DFE490(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 1688LL);
  if ( v1 == sub_DF5C30 )
    return 0;
  else
    return v1();
}
