// Function: sub_DFE120
// Address: 0xdfe120
//
__int64 __fastcall sub_DFE120(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 1472LL);
  if ( v1 == sub_DF62A0 )
    return 0;
  else
    return v1();
}
