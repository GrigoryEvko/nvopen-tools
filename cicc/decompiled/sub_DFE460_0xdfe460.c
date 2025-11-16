// Function: sub_DFE460
// Address: 0xdfe460
//
__int64 __fastcall sub_DFE460(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 1680LL);
  if ( v1 == sub_DF6320 )
    return 0;
  else
    return v1();
}
