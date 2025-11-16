// Function: sub_DFE670
// Address: 0xdfe670
//
__int64 __fastcall sub_DFE670(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 1640LL);
  if ( v1 == sub_DF6300 )
    return 0;
  else
    return v1();
}
