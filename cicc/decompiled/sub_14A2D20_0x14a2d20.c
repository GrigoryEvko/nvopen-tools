// Function: sub_14A2D20
// Address: 0x14a2d20
//
__int64 __fastcall sub_14A2D20(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 288LL);
  if ( v1 == sub_14A08C0 )
    return 1;
  else
    return v1();
}
