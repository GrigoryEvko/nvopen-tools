// Function: sub_DFE180
// Address: 0xdfe180
//
__int64 __fastcall sub_DFE180(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 1488LL);
  if ( v1 == sub_DF5CC0 )
    return 1;
  else
    return v1();
}
