// Function: sub_14A2C50
// Address: 0x14a2c50
//
__int64 __fastcall sub_14A2C50(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 256LL);
  if ( v1 == sub_14A0880 )
    return 1;
  else
    return v1();
}
