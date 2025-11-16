// Function: sub_DFD270
// Address: 0xdfd270
//
__int64 __fastcall sub_DFD270(__int64 a1, int a2, int a3)
{
  __int64 (*v3)(void); // rax

  v3 = *(__int64 (**)(void))(**(_QWORD **)a1 + 1216LL);
  if ( (char *)v3 == (char *)sub_DF6130 )
    return a3 == 0 || a2 != 55;
  else
    return v3();
}
