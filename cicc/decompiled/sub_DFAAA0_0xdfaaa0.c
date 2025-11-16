// Function: sub_DFAAA0
// Address: 0xdfaaa0
//
char __fastcall sub_DFAAA0(__int64 a1, __int64 a2, int a3)
{
  __int64 (*v3)(void); // rax

  v3 = *(__int64 (**)(void))(**(_QWORD **)a1 + 784LL);
  if ( (char *)v3 == (char *)sub_DF5EC0 )
    return a3 == 0;
  else
    return v3();
}
