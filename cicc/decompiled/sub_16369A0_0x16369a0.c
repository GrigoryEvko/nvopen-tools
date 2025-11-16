// Function: sub_16369A0
// Address: 0x16369a0
//
__int64 __fastcall sub_16369A0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rax

  v2 = sub_163A1D0(a1, a2);
  v3 = sub_163A340(v2, a1);
  if ( v3 )
    return (*(__int64 (**)(void))(v3 + 72))();
  else
    return 0;
}
