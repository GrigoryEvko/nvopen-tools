// Function: sub_14A3940
// Address: 0x14a3940
//
__int64 __fastcall sub_14A3940(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 824LL);
  if ( v1 == sub_14A0B20 )
    return 1;
  else
    return v1();
}
