// Function: sub_DFA630
// Address: 0xdfa630
//
__int64 __fastcall sub_DFA630(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 600LL);
  if ( v1 == sub_DF5E10 )
    return 0;
  else
    return v1();
}
