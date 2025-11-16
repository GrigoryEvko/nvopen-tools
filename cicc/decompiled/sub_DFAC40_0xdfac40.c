// Function: sub_DFAC40
// Address: 0xdfac40
//
__int64 __fastcall sub_DFAC40(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 848LL);
  if ( v1 == sub_DF5C80 )
    return 1;
  else
    return v1();
}
