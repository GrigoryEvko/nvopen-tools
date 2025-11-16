// Function: sub_14A3110
// Address: 0x14a3110
//
__int64 __fastcall sub_14A3110(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 472LL);
  if ( v1 == sub_14A0950 )
    return 0;
  else
    return v1();
}
