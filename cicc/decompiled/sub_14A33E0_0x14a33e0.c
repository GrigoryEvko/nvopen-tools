// Function: sub_14A33E0
// Address: 0x14a33e0
//
__int64 __fastcall sub_14A33E0(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 608LL);
  if ( v1 == sub_14A0A00 )
    return 1;
  else
    return v1();
}
