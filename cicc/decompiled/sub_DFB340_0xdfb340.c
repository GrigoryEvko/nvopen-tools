// Function: sub_DFB340
// Address: 0xdfb340
//
__int64 __fastcall sub_DFB340(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 1064LL);
  if ( v1 == sub_DF6070 )
    return 0;
  else
    return v1();
}
