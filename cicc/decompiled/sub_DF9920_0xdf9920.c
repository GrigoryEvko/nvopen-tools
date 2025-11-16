// Function: sub_DF9920
// Address: 0xdf9920
//
__int64 __fastcall sub_DF9920(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 216LL);
  if ( v1 == sub_DF5CB0 )
    return 1;
  else
    return v1();
}
