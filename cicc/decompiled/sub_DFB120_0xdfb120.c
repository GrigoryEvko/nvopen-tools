// Function: sub_DFB120
// Address: 0xdfb120
//
__int64 __fastcall sub_DFB120(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 976LL);
  if ( v1 == sub_DF5FC0 )
    return 8;
  else
    return v1();
}
