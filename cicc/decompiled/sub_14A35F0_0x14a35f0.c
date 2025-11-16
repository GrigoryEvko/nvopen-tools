// Function: sub_14A35F0
// Address: 0x14a35f0
//
__int64 __fastcall sub_14A35F0(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 712LL);
  if ( v1 == sub_14A0AC0 )
    return 0;
  else
    return v1();
}
