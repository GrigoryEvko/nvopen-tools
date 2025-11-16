// Function: sub_DFA8C0
// Address: 0xdfa8c0
//
__int64 __fastcall sub_DFA8C0(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 704LL);
  if ( v1 == sub_DF5CE0 )
    return 0;
  else
    return v1();
}
