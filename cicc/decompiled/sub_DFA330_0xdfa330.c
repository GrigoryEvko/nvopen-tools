// Function: sub_DFA330
// Address: 0xdfa330
//
__int64 __fastcall sub_DFA330(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 512LL);
  if ( v1 == sub_DF5E00 )
    return 2;
  else
    return v1();
}
