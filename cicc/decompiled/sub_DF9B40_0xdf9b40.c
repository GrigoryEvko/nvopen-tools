// Function: sub_DF9B40
// Address: 0xdf9b40
//
char __fastcall sub_DF9B40(__int64 a1, int a2)
{
  __int64 (*v2)(void); // rax

  v2 = *(__int64 (**)(void))(**(_QWORD **)a1 + 304LL);
  if ( (char *)v2 == (char *)sub_DF5D10 )
    return a2 == 0;
  else
    return v2();
}
