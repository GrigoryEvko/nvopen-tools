// Function: sub_27AC440
// Address: 0x27ac440
//
void __fastcall sub_27AC440(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rdx

  *a1 = a1 + 2;
  a1[1] = 0x400000000LL;
  v6 = *(unsigned int *)(a2 + 8);
  if ( (_DWORD)v6 )
    sub_27ABF90((__int64)a1, a2, v6, a4, a5, a6);
  a1[6] = a1 + 8;
  a1[7] = 0x400000000LL;
  if ( *(_DWORD *)(a2 + 56) )
    sub_27AC1D0((__int64)(a1 + 6), a2 + 48, v6, a4, a5, a6);
}
