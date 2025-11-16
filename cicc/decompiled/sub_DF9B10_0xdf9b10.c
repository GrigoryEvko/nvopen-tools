// Function: sub_DF9B10
// Address: 0xdf9b10
//
__int64 __fastcall sub_DF9B10(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 296LL);
  if ( v1 == sub_DF5C40 )
    return 0;
  else
    return v1();
}
