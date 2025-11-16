// Function: sub_C44320
// Address: 0xc44320
//
unsigned __int64 __fastcall sub_C44320(unsigned __int64 *a1, int a2, unsigned int a3)
{
  unsigned __int64 v3; // rax
  unsigned __int64 v4; // r8
  __int64 v5; // rsi
  __int64 v6; // rdi

  v3 = 0;
  if ( a2 )
    v3 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)a2);
  v4 = *a1;
  if ( *((_DWORD *)a1 + 2) <= 0x40u )
    return (v4 >> a3) & v3;
  v5 = (a2 + a3 - 1) >> 6;
  v6 = *(_QWORD *)(v4 + 8LL * (a3 >> 6)) >> a3;
  if ( (_DWORD)v5 == a3 >> 6 )
    return v6 & v3;
  else
    return (v6 | (*(_QWORD *)(v4 + 8 * v5) << (64 - (a3 & 0x3F)))) & v3;
}
