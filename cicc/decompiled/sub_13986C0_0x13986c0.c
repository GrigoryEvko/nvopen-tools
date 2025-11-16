// Function: sub_13986C0
// Address: 0x13986c0
//
__int64 __fastcall sub_13986C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int64 v4; // rsi
  __int64 v7; // rbx
  __int64 v8; // rax
  unsigned __int64 v9; // r14
  __int64 result; // rax

  v4 = a2 & 0xFFFFFFFFFFFFFFF8LL;
  v7 = *(_QWORD *)(a1 + 8);
  v8 = v7 + 32;
  if ( *(_QWORD *)(v7 + 16) != v4 )
  {
    do
    {
      v7 = v8;
      v8 += 32;
    }
    while ( *(_QWORD *)(v8 - 16) != v4 );
  }
  v9 = a3 & 0xFFFFFFFFFFFFFFF8LL;
  --*(_DWORD *)(*(_QWORD *)(v7 + 24) + 32LL);
  result = *(_QWORD *)(v7 + 16);
  if ( (a3 & 0xFFFFFFFFFFFFFFF8LL) != result )
  {
    if ( result != -8 && result != 0 && result != -16 )
      result = sub_1649B30(v7);
    *(_QWORD *)(v7 + 16) = v9;
    if ( (a3 & 0xFFFFFFFFFFFFFFF0LL) != 0xFFFFFFFFFFFFFFF0LL && v9 )
      result = sub_164C220(v7);
  }
  *(_QWORD *)(v7 + 24) = a4;
  ++*(_DWORD *)(a4 + 32);
  return result;
}
