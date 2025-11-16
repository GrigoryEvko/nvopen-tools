// Function: sub_E105D0
// Address: 0xe105d0
//
unsigned __int64 __fastcall sub_E105D0(__int64 a1, __int64 a2)
{
  _BYTE *v2; // r12
  __int64 v4; // rsi
  unsigned __int64 result; // rax
  void *v6; // rdi
  unsigned __int64 v7; // rsi
  unsigned __int64 v8; // rax

  v2 = *(_BYTE **)(a1 + 16);
  (*(void (__fastcall **)(_BYTE *))(*(_QWORD *)v2 + 32LL))(v2);
  if ( (v2[9] & 0xC0) != 0x40 )
    (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v2 + 40LL))(v2, a2);
  v4 = *(_QWORD *)(a2 + 8);
  result = *(_QWORD *)(a2 + 16);
  v6 = *(void **)a2;
  if ( v4 + 1 > result )
  {
    v7 = v4 + 993;
    v8 = 2 * result;
    if ( v7 > v8 )
      *(_QWORD *)(a2 + 16) = v7;
    else
      *(_QWORD *)(a2 + 16) = v8;
    result = realloc(v6);
    *(_QWORD *)a2 = result;
    v6 = (void *)result;
    if ( !result )
      abort();
    v4 = *(_QWORD *)(a2 + 8);
  }
  *((_BYTE *)v6 + v4) = 32;
  ++*(_QWORD *)(a2 + 8);
  return result;
}
