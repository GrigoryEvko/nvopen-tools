// Function: sub_DF96E0
// Address: 0xdf96e0
//
__int64 __fastcall sub_DF96E0(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 136LL);
  if ( v1 == sub_DF5C20 )
    return 0;
  else
    return v1();
}
