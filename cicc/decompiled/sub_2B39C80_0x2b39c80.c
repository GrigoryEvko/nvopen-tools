// Function: sub_2B39C80
// Address: 0x2b39c80
//
void __fastcall sub_2B39C80(__int64 a1, char **a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0x400000000LL;
  if ( *((_DWORD *)a2 + 2) )
    sub_2B0D510(a1, a2, a3, a4, a5, a6);
}
