// Function: sub_DFE340
// Address: 0xdfe340
//
__int64 __fastcall sub_DFE340(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 1560LL);
  if ( v1 == sub_DF62A0 )
    return 0;
  else
    return v1();
}
