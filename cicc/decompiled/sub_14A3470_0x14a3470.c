// Function: sub_14A3470
// Address: 0x14a3470
//
__int64 __fastcall sub_14A3470(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 632LL);
  if ( v1 == sub_14A0A20 )
    return 1;
  else
    return v1();
}
