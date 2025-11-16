// Function: sub_E452B0
// Address: 0xe452b0
//
__int64 __fastcall sub_E452B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v6; // rsi
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // r12
  __int64 result; // rax

  v6 = (_QWORD *)(a2 + 48);
  v7 = *v6 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (_QWORD *)v7 == v6 )
  {
    v8 = 0;
  }
  else
  {
    if ( !v7 )
      BUG();
    v8 = v7 - 24;
    if ( (unsigned int)*(unsigned __int8 *)(v7 - 24) - 30 > 0xA )
      v8 = 0;
  }
  result = *(unsigned int *)(a1 + 8);
  if ( result + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
  {
    sub_C8D5F0(a1, (const void *)(a1 + 16), result + 1, 8u, a5, a6);
    result = *(unsigned int *)(a1 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a1 + 8 * result) = v8;
  ++*(_DWORD *)(a1 + 8);
  return result;
}
