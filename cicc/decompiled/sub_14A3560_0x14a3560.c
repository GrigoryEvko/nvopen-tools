// Function: sub_14A3560
// Address: 0x14a3560
//
__int64 __fastcall sub_14A3560(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 688LL);
  if ( v1 == sub_14A0A90 )
    return 1;
  else
    return v1();
}
