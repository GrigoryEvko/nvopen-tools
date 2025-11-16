// Function: sub_39DFE80
// Address: 0x39dfe80
//
__int64 __fastcall sub_39DFE80(__int64 a1)
{
  unsigned __int64 v2; // r12
  __int64 v3; // r13
  char *v4; // rsi
  size_t v5; // rdx
  void *v6; // rdi
  __int64 result; // rax

  v2 = *(unsigned int *)(a1 + 312);
  if ( *(_DWORD *)(a1 + 312) )
  {
    v3 = *(_QWORD *)(a1 + 272);
    v4 = *(char **)(a1 + 304);
    v5 = *(unsigned int *)(a1 + 312);
    v6 = *(void **)(v3 + 24);
    if ( v2 > *(_QWORD *)(v3 + 16) - (_QWORD)v6 )
    {
      result = sub_16E7EE0(v3, v4, v5);
    }
    else
    {
      result = (__int64)memcpy(v6, v4, v5);
      *(_QWORD *)(v3 + 24) += v2;
    }
  }
  *(_DWORD *)(a1 + 312) = 0;
  return result;
}
