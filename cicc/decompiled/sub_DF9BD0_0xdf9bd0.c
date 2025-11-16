// Function: sub_DF9BD0
// Address: 0xdf9bd0
//
__int64 __fastcall sub_DF9BD0(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 328LL);
  if ( v1 == sub_DF5D30 )
    return 0;
  else
    return v1();
}
