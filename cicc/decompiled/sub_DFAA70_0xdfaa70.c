// Function: sub_DFAA70
// Address: 0xdfaa70
//
char __fastcall sub_DFAA70(__int64 a1, __int64 a2, int a3)
{
  __int64 (*v3)(void); // rax

  v3 = *(__int64 (**)(void))(**(_QWORD **)a1 + 776LL);
  if ( (char *)v3 == (char *)sub_DF5EB0 )
    return a3 == -1;
  else
    return v3();
}
