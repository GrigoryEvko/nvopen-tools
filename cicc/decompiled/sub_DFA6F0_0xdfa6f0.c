// Function: sub_DFA6F0
// Address: 0xdfa6f0
//
__int64 __fastcall sub_DFA6F0(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 640LL);
  if ( v1 == sub_DF5CE0 )
    return 0;
  else
    return v1();
}
