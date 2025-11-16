// Function: sub_DFB690
// Address: 0xdfb690
//
__int64 __fastcall sub_DFB690(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 1144LL);
  if ( v1 == sub_DF5CE0 )
    return 0;
  else
    return v1();
}
