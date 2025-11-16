// Function: sub_DFD220
// Address: 0xdfd220
//
__int64 __fastcall sub_DFD220(__int64 a1)
{
  __int64 (*v1)(void); // r9

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 1208LL);
  if ( v1 == sub_DF6120 )
    return 1;
  else
    return v1();
}
