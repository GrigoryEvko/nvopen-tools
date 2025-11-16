// Function: sub_DF9880
// Address: 0xdf9880
//
__int64 __fastcall sub_DF9880(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 192LL);
  if ( v1 == sub_DF5C70 )
    return 0;
  else
    return v1();
}
