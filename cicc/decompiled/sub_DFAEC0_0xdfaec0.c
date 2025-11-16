// Function: sub_DFAEC0
// Address: 0xdfaec0
//
__int64 __fastcall sub_DFAEC0(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 896LL);
  if ( v1 == sub_DF5F60 )
    return 0;
  else
    return v1();
}
