// Function: sub_33CC630
// Address: 0x33cc630
//
unsigned __int64 __fastcall sub_33CC630(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v6; // rax
  __int64 *v7; // rsi
  unsigned __int64 v8; // rcx
  __int64 v9; // rdx
  unsigned __int64 result; // rax
  __int64 v11; // rsi

  v6 = *(_QWORD **)(a1 + 408);
  v7 = (__int64 *)v6[1];
  v8 = *v6 & 0xFFFFFFFFFFFFFFF8LL;
  v9 = v8 | *v7 & 7;
  *v7 = v9;
  *(_QWORD *)(v8 + 8) = v7;
  v6[1] = 0;
  *v6 &= 7uLL;
  for ( result = *(_QWORD *)(a1 + 400) & 0xFFFFFFFFFFFFFFF8LL;
        a1 + 400 != result;
        result = *(_QWORD *)(a1 + 400) & 0xFFFFFFFFFFFFFFF8LL )
  {
    v11 = *(_QWORD *)(a1 + 408);
    if ( v11 )
      v11 -= 8;
    sub_33CC1B0(a1, v11, v9, v8, a5, a6);
  }
  return result;
}
