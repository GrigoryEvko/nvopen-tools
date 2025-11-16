// Function: sub_DFAB60
// Address: 0xdfab60
//
__int64 __fastcall sub_DFAB60(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 824LL);
  if ( v1 == sub_DF5CC0 )
    return 1;
  else
    return v1();
}
