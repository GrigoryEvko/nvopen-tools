// Function: sub_DFB150
// Address: 0xdfb150
//
__int64 __fastcall sub_DFB150(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(**(_QWORD **)a1 + 984LL);
  if ( v1 == sub_DF5C30 )
    return 0;
  else
    return v1();
}
