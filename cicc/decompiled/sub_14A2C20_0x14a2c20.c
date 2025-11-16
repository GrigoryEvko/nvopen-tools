// Function: sub_14A2C20
// Address: 0x14a2c20
//
__int64 __fastcall sub_14A2C20(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 248LL);
  if ( v1 == sub_14A0870 )
    return 0;
  else
    return v1();
}
