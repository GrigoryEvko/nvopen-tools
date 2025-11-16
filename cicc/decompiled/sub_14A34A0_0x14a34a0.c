// Function: sub_14A34A0
// Address: 0x14a34a0
//
__int64 __fastcall sub_14A34A0(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 640LL);
  if ( v1 == sub_14A0A30 )
    return 1;
  else
    return v1();
}
