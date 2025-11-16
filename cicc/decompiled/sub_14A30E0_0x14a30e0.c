// Function: sub_14A30E0
// Address: 0x14a30e0
//
__int64 __fastcall sub_14A30E0(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 464LL);
  if ( v1 == sub_14A0950 )
    return 0;
  else
    return v1();
}
