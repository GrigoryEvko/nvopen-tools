// Function: sub_14A26B0
// Address: 0x14a26b0
//
__int64 __fastcall sub_14A26B0(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 72LL);
  if ( v1 == sub_14A07D0 )
    return 1;
  else
    return v1();
}
