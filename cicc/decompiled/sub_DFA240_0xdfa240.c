// Function: sub_DFA240
// Address: 0xdfa240
//
__int64 __fastcall sub_DFA240(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 472LL);
  if ( v1 == sub_DF5C80 )
    return 1;
  else
    return v1();
}
