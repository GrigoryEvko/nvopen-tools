// Function: sub_DF93E0
// Address: 0xdf93e0
//
__int64 __fastcall sub_DF93E0(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 48LL);
  if ( v1 == sub_DF5B90 )
    return 8;
  else
    return v1();
}
