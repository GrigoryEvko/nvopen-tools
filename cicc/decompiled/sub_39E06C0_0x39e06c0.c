// Function: sub_39E06C0
// Address: 0x39e06c0
//
_BYTE *__fastcall sub_39E06C0(__int64 a1)
{
  unsigned __int64 v2; // r13
  _BYTE *result; // rax
  __int64 v4; // rdi
  __int64 v5; // r14
  char *v6; // rsi
  size_t v7; // rdx
  void *v8; // rdi

  v2 = *(unsigned int *)(a1 + 312);
  if ( *(_DWORD *)(a1 + 312) )
  {
    v5 = *(_QWORD *)(a1 + 272);
    v6 = *(char **)(a1 + 304);
    v7 = *(unsigned int *)(a1 + 312);
    v8 = *(void **)(v5 + 24);
    if ( v2 > *(_QWORD *)(v5 + 16) - (_QWORD)v8 )
    {
      sub_16E7EE0(v5, v6, v7);
    }
    else
    {
      memcpy(v8, v6, v7);
      *(_QWORD *)(v5 + 24) += v2;
    }
  }
  *(_DWORD *)(a1 + 312) = 0;
  if ( (*(_BYTE *)(a1 + 680) & 1) != 0 )
    return sub_39E0440(a1);
  v4 = *(_QWORD *)(a1 + 272);
  result = *(_BYTE **)(v4 + 24);
  if ( (unsigned __int64)result >= *(_QWORD *)(v4 + 16) )
    return (_BYTE *)sub_16E7DE0(v4, 10);
  *(_QWORD *)(v4 + 24) = result + 1;
  *result = 10;
  return result;
}
