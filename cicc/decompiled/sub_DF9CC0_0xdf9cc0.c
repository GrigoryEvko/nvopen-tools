// Function: sub_DF9CC0
// Address: 0xdf9cc0
//
__int64 __fastcall sub_DF9CC0(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 392LL);
  if ( v1 == sub_DF5C30 )
    return 0;
  else
    return v1();
}
