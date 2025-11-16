// Function: sub_DFE700
// Address: 0xdfe700
//
__int64 __fastcall sub_DFE700(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 1704LL);
  if ( v1 == sub_DF6330 )
    return 0;
  else
    return v1();
}
