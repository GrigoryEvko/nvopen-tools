// Function: sub_DFAE60
// Address: 0xdfae60
//
__int64 __fastcall sub_DFAE60(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 880LL);
  if ( v1 == sub_DF5CE0 )
    return 0;
  else
    return v1();
}
