// Function: sub_14A38B0
// Address: 0x14a38b0
//
__int64 __fastcall sub_14A38B0(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 800LL);
  if ( v1 == sub_14A0B10 )
    return 128;
  else
    return v1();
}
