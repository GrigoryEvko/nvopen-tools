// Function: sub_DFA980
// Address: 0xdfa980
//
__int64 __fastcall sub_DFA980(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 736LL);
  if ( v1 == sub_DF5CC0 )
    return 1;
  else
    return v1();
}
