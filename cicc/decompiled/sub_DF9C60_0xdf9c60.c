// Function: sub_DF9C60
// Address: 0xdf9c60
//
__int64 __fastcall sub_DF9C60(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 376LL);
  if ( v1 == sub_DF5D80 )
    return 0;
  else
    return v1();
}
