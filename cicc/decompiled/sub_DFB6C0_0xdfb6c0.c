// Function: sub_DFB6C0
// Address: 0xdfb6c0
//
char __fastcall sub_DFB6C0(__int64 a1, int a2)
{
  __int64 (*v2)(void); // rax

  v2 = *(__int64 (**)(void))(**(_QWORD **)a1 + 1152LL);
  if ( (char *)v2 == (char *)sub_DF5D10 )
    return a2 == 0;
  else
    return v2();
}
