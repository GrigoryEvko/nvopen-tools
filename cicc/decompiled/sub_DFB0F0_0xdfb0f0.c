// Function: sub_DFB0F0
// Address: 0xdfb0f0
//
__int64 __fastcall sub_DFB0F0(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 968LL);
  if ( v1 == sub_DF5E40 )
    return 0;
  else
    return v1();
}
