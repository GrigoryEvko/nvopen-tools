// Function: sub_14A3200
// Address: 0x14a3200
//
__int64 __fastcall sub_14A3200(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 512LL);
  if ( v1 == sub_14A0990 )
    return 0;
  else
    return v1();
}
