// Function: sub_DFAF20
// Address: 0xdfaf20
//
__int64 __fastcall sub_DFAF20(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 920LL);
  if ( v1 == sub_DF5CC0 )
    return 1;
  else
    return v1();
}
