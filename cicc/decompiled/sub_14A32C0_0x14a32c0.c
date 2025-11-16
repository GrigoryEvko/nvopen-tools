// Function: sub_14A32C0
// Address: 0x14a32c0
//
__int64 __fastcall sub_14A32C0(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 560LL);
  if ( v1 == sub_14A07D0 )
    return 1;
  else
    return v1();
}
