// Function: sub_DFB0A0
// Address: 0xdfb0a0
//
__int64 __fastcall sub_DFB0A0(__int64 a1)
{
  __int64 (*v1)(void); // r10

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 960LL);
  if ( v1 == sub_DF5FB0 )
    return 0;
  else
    return v1();
}
