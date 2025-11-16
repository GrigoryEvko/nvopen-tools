// Function: sub_DF9850
// Address: 0xdf9850
//
__int64 __fastcall sub_DF9850(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 184LL);
  if ( v1 == sub_DF5C60 )
    return 0xFFFFFFFFLL;
  else
    return v1();
}
