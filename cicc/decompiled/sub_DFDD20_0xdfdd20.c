// Function: sub_DFDD20
// Address: 0xdfdd20
//
__int64 __fastcall sub_DFDD20(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 1392LL);
  if ( v1 == sub_DF6270 )
    return 0;
  else
    return v1();
}
