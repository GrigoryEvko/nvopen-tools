// Function: sub_DFA300
// Address: 0xdfa300
//
__int64 __fastcall sub_DFA300(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 504LL);
  if ( v1 == sub_DF5DF0 )
    return 0;
  else
    return v1();
}
