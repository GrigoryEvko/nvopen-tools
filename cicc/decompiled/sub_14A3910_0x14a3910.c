// Function: sub_14A3910
// Address: 0x14a3910
//
__int64 __fastcall sub_14A3910(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 816LL);
  if ( v1 == sub_14A08C0 )
    return 1;
  else
    return v1();
}
