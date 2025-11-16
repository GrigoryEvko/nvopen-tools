// Function: sub_DFE580
// Address: 0xdfe580
//
__int64 __fastcall sub_DFE580(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 1600LL);
  if ( v1 == sub_DF62F0 )
    return 0;
  else
    return v1();
}
