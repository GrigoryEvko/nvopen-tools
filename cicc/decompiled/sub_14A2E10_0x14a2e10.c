// Function: sub_14A2E10
// Address: 0x14a2e10
//
__int64 __fastcall sub_14A2E10(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 344LL);
  if ( v1 == sub_14A0810 )
    return 0;
  else
    return v1();
}
