// Function: sub_DFD440
// Address: 0xdfd440
//
__int64 __fastcall sub_DFD440(__int64 a1, int a2, int a3)
{
  __int64 (*v3)(void); // rcx
  __int64 v4; // r8

  v3 = *(__int64 (**)(void))(**(_QWORD **)a1 + 1264LL);
  if ( (char *)v3 != (char *)sub_DF6510 )
    return v3();
  v4 = 0;
  if ( a2 == 65 )
    return -(__int64)(a3 == 0) | 1;
  return v4;
}
