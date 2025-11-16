// Function: sub_DFA540
// Address: 0xdfa540
//
__int64 __fastcall sub_DFA540(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 632LL);
  if ( v1 == sub_DF5E50 )
    return 0;
  else
    return v1();
}
