// Function: sub_DFD500
// Address: 0xdfd500
//
__int64 __fastcall sub_DFD500(__int64 a1)
{
  __int64 (*v1)(void); // r10

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 1288LL);
  if ( v1 == sub_DF61C0 )
    return 1;
  else
    return v1();
}
