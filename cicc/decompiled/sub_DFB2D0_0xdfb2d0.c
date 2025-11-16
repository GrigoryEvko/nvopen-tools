// Function: sub_DFB2D0
// Address: 0xdfb2d0
//
__int64 __fastcall sub_DFB2D0(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 1048LL);
  if ( v1 == sub_DF5CF0 )
    return 0;
  else
    return v1();
}
