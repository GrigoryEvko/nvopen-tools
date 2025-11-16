// Function: sub_14A34D0
// Address: 0x14a34d0
//
__int64 __fastcall sub_14A34D0(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 648LL);
  if ( v1 == sub_14A0A40 )
    return 1;
  else
    return v1();
}
