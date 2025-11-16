// Function: sub_DFDC70
// Address: 0xdfdc70
//
__int64 __fastcall sub_DFDC70(__int64 a1)
{
  __int64 (*v1)(void); // r9

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 1336LL);
  if ( v1 == sub_DF6220 )
    return 1;
  else
    return v1();
}
