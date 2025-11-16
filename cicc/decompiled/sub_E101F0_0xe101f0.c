// Function: sub_E101F0
// Address: 0xe101f0
//
_BYTE *__fastcall sub_E101F0(__int64 a1, __int64 a2)
{
  __int64 v3; // rsi
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // rsi
  unsigned __int64 v6; // rax
  _BYTE *result; // rax

  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 16) + 32LL))(*(_QWORD *)(a1 + 16));
  v3 = *(_QWORD *)(a2 + 8);
  v4 = *(_QWORD *)(a2 + 16);
  if ( v3 + 1 <= v4 )
  {
    result = *(_BYTE **)a2;
    *(_BYTE *)(*(_QWORD *)a2 + v3) = 32;
    ++*(_QWORD *)(a2 + 8);
  }
  else
  {
    v5 = v3 + 993;
    v6 = 2 * v4;
    if ( v5 > v6 )
      *(_QWORD *)(a2 + 16) = v5;
    else
      *(_QWORD *)(a2 + 16) = v6;
    result = (_BYTE *)realloc(*(void **)a2);
    *(_QWORD *)a2 = result;
    if ( !result )
      abort();
    result[(*(_QWORD *)(a2 + 8))++] = 32;
  }
  return result;
}
