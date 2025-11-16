// Function: sub_DFAA10
// Address: 0xdfaa10
//
__int64 __fastcall sub_DFAA10(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 760LL);
  if ( v1 == sub_DF5CF0 )
    return 0;
  else
    return v1();
}
