// Function: sub_DF9820
// Address: 0xdf9820
//
__int64 __fastcall sub_DF9820(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 176LL);
  if ( v1 == sub_DF5C50 )
    return 1;
  else
    return v1();
}
