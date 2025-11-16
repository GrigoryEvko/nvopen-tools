// Function: sub_14A3380
// Address: 0x14a3380
//
__int64 __fastcall sub_14A3380(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 592LL);
  if ( v1 == sub_14A09E0 )
    return 1;
  else
    return v1();
}
