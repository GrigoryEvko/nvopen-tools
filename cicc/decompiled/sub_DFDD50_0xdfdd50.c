// Function: sub_DFDD50
// Address: 0xdfdd50
//
__int64 __fastcall sub_DFDD50(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 1400LL);
  if ( v1 == sub_DF5E40 )
    return 0;
  else
    return v1();
}
