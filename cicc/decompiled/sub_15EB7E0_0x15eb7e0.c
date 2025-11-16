// Function: sub_15EB7E0
// Address: 0x15eb7e0
//
void __fastcall sub_15EB7E0(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rdx
  __int64 v7; // rdi
  __int64 v8; // rsi

  if ( a2 < *(_DWORD *)(a1 + 72) )
  {
    *(_DWORD *)(a1 + 12) = a2;
    v6 = *(_QWORD *)(a1 + 64);
    v7 = a1 + 16;
    v8 = v6 + 56LL * a2;
    *(_DWORD *)(v7 - 12) = *(_DWORD *)v8;
    sub_15EB610(v7, (__int64 *)(v8 + 8), v6, a4, a5, a6);
  }
}
