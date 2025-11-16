// Function: sub_DFA0F0
// Address: 0xdfa0f0
//
__int64 __fastcall sub_DFA0F0(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 440LL);
  if ( v1 == sub_DF5DD0 )
    return 0;
  else
    return v1();
}
