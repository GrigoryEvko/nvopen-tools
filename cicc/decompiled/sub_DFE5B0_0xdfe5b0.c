// Function: sub_DFE5B0
// Address: 0xdfe5b0
//
__int64 __fastcall sub_DFE5B0(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 1608LL);
  if ( v1 == sub_DF5B80 )
    return 1;
  else
    return v1();
}
