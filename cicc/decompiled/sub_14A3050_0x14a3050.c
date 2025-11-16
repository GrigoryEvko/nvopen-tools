// Function: sub_14A3050
// Address: 0x14a3050
//
__int64 __fastcall sub_14A3050(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 440LL);
  if ( v1 == sub_14A0940 )
    return 1;
  else
    return v1();
}
