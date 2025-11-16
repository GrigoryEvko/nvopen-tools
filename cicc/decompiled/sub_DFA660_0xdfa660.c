// Function: sub_DFA660
// Address: 0xdfa660
//
__int64 __fastcall sub_DFA660(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 608LL);
  if ( v1 == sub_DF5E10 )
    return 0;
  else
    return v1();
}
