// Function: sub_DF93B0
// Address: 0xdf93b0
//
__int64 __fastcall sub_DF93B0(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 40LL);
  if ( v1 == sub_DF5B80 )
    return 1;
  else
    return v1();
}
