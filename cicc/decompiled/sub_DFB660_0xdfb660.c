// Function: sub_DFB660
// Address: 0xdfb660
//
__int64 __fastcall sub_DFB660(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 1136LL);
  if ( v1 == sub_DF5C60 )
    return 0xFFFFFFFFLL;
  else
    return v1();
}
