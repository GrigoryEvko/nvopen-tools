// Function: sub_DFD3F0
// Address: 0xdfd3f0
//
__int64 __fastcall sub_DFD3F0(__int64 a1)
{
  __int64 (*v1)(void); // r9

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 1248LL);
  if ( v1 == sub_DF6180 )
    return 1;
  else
    return v1();
}
