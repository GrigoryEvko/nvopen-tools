// Function: sub_DFE4C0
// Address: 0xdfe4c0
//
__int64 __fastcall sub_DFE4C0(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 1696LL);
  if ( v1 == sub_DF5C60 )
    return 0xFFFFFFFFLL;
  else
    return v1();
}
