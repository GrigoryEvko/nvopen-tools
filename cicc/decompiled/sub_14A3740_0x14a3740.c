// Function: sub_14A3740
// Address: 0x14a3740
//
__int64 __fastcall sub_14A3740(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 752LL);
  if ( v1 == sub_14A0AF0 )
    return 0;
  else
    return v1();
}
