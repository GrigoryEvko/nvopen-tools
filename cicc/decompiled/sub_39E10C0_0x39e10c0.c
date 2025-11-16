// Function: sub_39E10C0
// Address: 0x39e10c0
//
_BYTE *__fastcall sub_39E10C0(__int64 a1)
{
  __int64 v2; // rdi
  void *v3; // rdx
  unsigned __int64 v4; // r13
  _BYTE *result; // rax
  __int64 v6; // rdi
  __int64 v7; // r14
  char *v8; // rsi
  size_t v9; // rdx
  void *v10; // rdi

  v2 = *(_QWORD *)(a1 + 272);
  v3 = *(void **)(v2 + 24);
  if ( *(_QWORD *)(v2 + 16) - (_QWORD)v3 <= 0xEu )
  {
    sub_16E7EE0(v2, "\t.bundle_unlock", 0xFu);
  }
  else
  {
    qmemcpy(v3, "\t.bundle_unlock", 15);
    *(_QWORD *)(v2 + 24) += 15LL;
  }
  v4 = *(unsigned int *)(a1 + 312);
  if ( *(_DWORD *)(a1 + 312) )
  {
    v7 = *(_QWORD *)(a1 + 272);
    v8 = *(char **)(a1 + 304);
    v9 = *(unsigned int *)(a1 + 312);
    v10 = *(void **)(v7 + 24);
    if ( v4 > *(_QWORD *)(v7 + 16) - (_QWORD)v10 )
    {
      sub_16E7EE0(*(_QWORD *)(a1 + 272), v8, v9);
    }
    else
    {
      memcpy(v10, v8, v9);
      *(_QWORD *)(v7 + 24) += v4;
    }
  }
  *(_DWORD *)(a1 + 312) = 0;
  if ( (*(_BYTE *)(a1 + 680) & 1) != 0 )
    return sub_39E0440(a1);
  v6 = *(_QWORD *)(a1 + 272);
  result = *(_BYTE **)(v6 + 24);
  if ( (unsigned __int64)result >= *(_QWORD *)(v6 + 16) )
    return (_BYTE *)sub_16E7DE0(v6, 10);
  *(_QWORD *)(v6 + 24) = result + 1;
  *result = 10;
  return result;
}
