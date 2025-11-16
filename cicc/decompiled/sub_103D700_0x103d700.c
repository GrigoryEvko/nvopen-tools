// Function: sub_103D700
// Address: 0x103d700
//
_BYTE *__fastcall sub_103D700(_BYTE *a1, __int64 a2)
{
  __int64 *v2; // rax
  void *v4; // rdx
  __int64 v5; // rbx
  unsigned __int64 v6; // rsi
  _BYTE *result; // rax
  void *v8; // rdx

  v2 = (__int64 *)(a1 - 64);
  if ( *a1 == 26 )
    v2 = (__int64 *)(a1 - 32);
  v4 = *(void **)(a2 + 32);
  v5 = *v2;
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v4 <= 9u )
  {
    sub_CB6200(a2, "MemoryUse(", 0xAu);
  }
  else
  {
    qmemcpy(v4, "MemoryUse(", 10);
    *(_QWORD *)(a2 + 32) += 10LL;
  }
  if ( v5 && (*(_BYTE *)v5 != 27 ? (v6 = *(unsigned int *)(v5 + 72)) : (v6 = *(unsigned int *)(v5 + 80)), (_DWORD)v6) )
  {
    sub_CB59D0(a2, v6);
    result = *(_BYTE **)(a2 + 32);
  }
  else
  {
    v8 = *(void **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v8 > 0xAu )
    {
      qmemcpy(v8, "liveOnEntry", 11);
      result = (_BYTE *)(*(_QWORD *)(a2 + 32) + 11LL);
      *(_QWORD *)(a2 + 32) = result;
      if ( *(_QWORD *)(a2 + 24) > (unsigned __int64)result )
        goto LABEL_11;
      return (_BYTE *)sub_CB5D20(a2, 41);
    }
    sub_CB6200(a2, "liveOnEntry", 0xBu);
    result = *(_BYTE **)(a2 + 32);
  }
  if ( *(_QWORD *)(a2 + 24) > (unsigned __int64)result )
  {
LABEL_11:
    *(_QWORD *)(a2 + 32) = result + 1;
    *result = 41;
    return result;
  }
  return (_BYTE *)sub_CB5D20(a2, 41);
}
