// Function: sub_DF9B70
// Address: 0xdf9b70
//
__int64 __fastcall sub_DF9B70(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 312LL);
  if ( v1 == sub_DF5D20 )
    return 0xFFFFFFFFLL;
  else
    return v1();
}
