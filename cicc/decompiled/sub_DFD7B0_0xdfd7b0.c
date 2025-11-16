// Function: sub_DFD7B0
// Address: 0xdfd7b0
//
__int64 __fastcall sub_DFD7B0(__int64 a1)
{
  __int64 (*v1)(void); // r10

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 1368LL);
  if ( v1 == sub_DF6250 )
    return 1;
  else
    return v1();
}
