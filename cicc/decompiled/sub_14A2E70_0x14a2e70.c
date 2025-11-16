// Function: sub_14A2E70
// Address: 0x14a2e70
//
__int64 __fastcall sub_14A2E70(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 360LL);
  if ( v1 == sub_14A08F0 )
    return 0;
  else
    return v1();
}
