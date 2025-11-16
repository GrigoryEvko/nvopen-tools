// Function: sub_DFAB00
// Address: 0xdfab00
//
__int64 __fastcall sub_DFAB00(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 800LL);
  if ( v1 == sub_DF5EE0 )
    return 0;
  else
    return v1();
}
