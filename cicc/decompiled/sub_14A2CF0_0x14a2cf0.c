// Function: sub_14A2CF0
// Address: 0x14a2cf0
//
__int64 __fastcall sub_14A2CF0(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 280LL);
  if ( v1 == sub_14A08B0 )
    return 0;
  else
    return v1();
}
