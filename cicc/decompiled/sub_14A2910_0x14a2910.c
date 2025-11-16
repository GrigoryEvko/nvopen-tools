// Function: sub_14A2910
// Address: 0x14a2910
//
__int64 __fastcall sub_14A2910(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 112LL);
  if ( v1 == sub_14A0800 )
    return 0;
  else
    return v1();
}
