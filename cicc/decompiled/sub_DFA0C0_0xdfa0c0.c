// Function: sub_DFA0C0
// Address: 0xdfa0c0
//
__int64 __fastcall sub_DFA0C0(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 432LL);
  if ( v1 == sub_DF5DD0 )
    return 0;
  else
    return v1();
}
