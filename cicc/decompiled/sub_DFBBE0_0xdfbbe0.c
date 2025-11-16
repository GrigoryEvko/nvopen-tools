// Function: sub_DFBBE0
// Address: 0xdfbbe0
//
__int64 __fastcall sub_DFBBE0(__int64 a1)
{
  __int64 (*v1)(void); // r10

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 1184LL);
  if ( v1 == sub_DF6100 )
    return 0;
  else
    return v1();
}
