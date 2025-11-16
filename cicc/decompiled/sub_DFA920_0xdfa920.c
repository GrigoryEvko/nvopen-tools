// Function: sub_DFA920
// Address: 0xdfa920
//
__int64 __fastcall sub_DFA920(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 720LL);
  if ( v1 == sub_DF5EA0 )
    return 1;
  else
    return v1();
}
