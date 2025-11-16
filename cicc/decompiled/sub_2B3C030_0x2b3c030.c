// Function: sub_2B3C030
// Address: 0x2b3c030
//
void __fastcall sub_2B3C030(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0x600000000LL;
  if ( *(_DWORD *)(a2 + 8) )
    sub_2B0CFB0(a1, a2, a3, a4, a5, a6);
}
