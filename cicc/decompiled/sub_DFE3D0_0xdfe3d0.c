// Function: sub_DFE3D0
// Address: 0xdfe3d0
//
__int64 __fastcall sub_DFE3D0(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 1584LL);
  if ( v1 == sub_DF5C80 )
    return 1;
  else
    return v1();
}
