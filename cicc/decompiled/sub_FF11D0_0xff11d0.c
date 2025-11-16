// Function: sub_FF11D0
// Address: 0xff11d0
//
void __fastcall sub_FF11D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0xC00000000LL;
  if ( *(_DWORD *)(a2 + 8) )
    sub_FEE1E0(a1, a2, a3, a4, a5, a6);
}
