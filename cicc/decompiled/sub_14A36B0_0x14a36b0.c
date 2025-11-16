// Function: sub_14A36B0
// Address: 0x14a36b0
//
__int64 __fastcall sub_14A36B0(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 728LL);
  if ( v1 == sub_14A0AE0 )
    return 0;
  else
    return v1();
}
