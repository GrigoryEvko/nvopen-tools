// Function: sub_DF9C00
// Address: 0xdf9c00
//
__int64 __fastcall sub_DF9C00(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 336LL);
  if ( v1 == sub_DF5D40 )
    return 0;
  else
    return v1();
}
