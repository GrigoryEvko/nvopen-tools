// Function: sub_DFAA40
// Address: 0xdfaa40
//
__int64 __fastcall sub_DFAA40(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 768LL);
  if ( v1 == sub_DF5C40 )
    return 0;
  else
    return v1();
}
