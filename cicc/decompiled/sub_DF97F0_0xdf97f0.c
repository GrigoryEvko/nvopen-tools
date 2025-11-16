// Function: sub_DF97F0
// Address: 0xdf97f0
//
__int64 __fastcall sub_DF97F0(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 168LL);
  if ( v1 == sub_DF5C40 )
    return 0;
  else
    return v1();
}
