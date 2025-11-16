// Function: sub_14A31A0
// Address: 0x14a31a0
//
__int64 __fastcall sub_14A31A0(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 496LL);
  if ( v1 == sub_14A0980 )
    return 128;
  else
    return v1();
}
