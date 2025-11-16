// Function: sub_DFDB60
// Address: 0xdfdb60
//
__int64 __fastcall sub_DFDB60(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 1376LL);
  if ( v1 == sub_DF5EA0 )
    return 1;
  else
    return v1();
}
