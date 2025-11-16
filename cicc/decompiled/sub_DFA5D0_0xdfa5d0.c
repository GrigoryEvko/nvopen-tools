// Function: sub_DFA5D0
// Address: 0xdfa5d0
//
__int64 __fastcall sub_DFA5D0(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 584LL);
  if ( v1 == sub_DF5E10 )
    return 0;
  else
    return v1();
}
