// Function: sub_14A29A0
// Address: 0x14a29a0
//
__int64 __fastcall sub_14A29A0(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 136LL);
  if ( v1 == sub_14A0820 )
    return 0xFFFFFFFFLL;
  else
    return v1();
}
