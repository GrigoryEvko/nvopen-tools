// Function: sub_DFA270
// Address: 0xdfa270
//
__int64 __fastcall sub_DFA270(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 480LL);
  if ( v1 == sub_DF5CE0 )
    return 0;
  else
    return v1();
}
