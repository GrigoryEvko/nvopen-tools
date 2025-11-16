// Function: sub_14A2B00
// Address: 0x14a2b00
//
__int64 __fastcall sub_14A2B00(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 200LL);
  if ( v1 == sub_14A0800 )
    return 0;
  else
    return v1();
}
