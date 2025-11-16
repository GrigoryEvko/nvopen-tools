// Function: sub_2B7C810
// Address: 0x2b7c810
//
__int64 ***__fastcall sub_2B7C810(__int64 a1, int *a2, unsigned __int64 a3, __int64 *a4, __int64 a5)
{
  __int64 v8; // rsi
  __int64 *v9; // rdi

  v8 = *a4;
  if ( a5 == 1 )
  {
    if ( *(_DWORD *)(*(_QWORD *)(v8 + 8) + 32LL) == a3 )
    {
      if ( (unsigned __int8)sub_B4ED80(a2, a3, a3) )
        return (__int64 ***)*a4;
      v8 = *a4;
    }
    return sub_2B7C3E0(*(__int64 **)(a1 + 8), v8, 0, (__int64)a2, a3);
  }
  v9 = *(__int64 **)(a1 + 8);
  if ( !v8 )
    v8 = *(_QWORD *)(*(_QWORD *)a1 - 96LL);
  return sub_2B7C3E0(v9, v8, a4[a5 - 1], (__int64)a2, a3);
}
