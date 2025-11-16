// Function: sub_1423540
// Address: 0x1423540
//
_BYTE *__fastcall sub_1423540(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  void *v3; // rdx
  __int64 v4; // rbx
  __int64 v5; // rsi
  _BYTE *result; // rax
  unsigned __int8 v7; // bl
  _BYTE *v8; // rax
  void *v9; // rdx

  v2 = a2;
  v3 = *(void **)(a2 + 24);
  v4 = *(_QWORD *)(a1 - 24);
  if ( *(_QWORD *)(a2 + 16) - (_QWORD)v3 <= 9u )
  {
    sub_16E7EE0(a2, "MemoryUse(", 10);
  }
  else
  {
    qmemcpy(v3, "MemoryUse(", 10);
    *(_QWORD *)(a2 + 24) += 10LL;
  }
  if ( v4 )
  {
    v5 = *(_BYTE *)(v4 + 16) == 22 ? *(unsigned int *)(v4 + 84) : *(unsigned int *)(v4 + 72);
    if ( (_DWORD)v5 )
    {
      sub_16E7A90(v2, v5);
      result = *(_BYTE **)(v2 + 24);
      goto LABEL_8;
    }
  }
  v9 = *(void **)(v2 + 24);
  if ( *(_QWORD *)(v2 + 16) - (_QWORD)v9 <= 0xAu )
  {
    sub_16E7EE0(v2, "liveOnEntry", 11);
    result = *(_BYTE **)(v2 + 24);
LABEL_8:
    if ( *(_QWORD *)(v2 + 16) > (unsigned __int64)result )
      goto LABEL_9;
LABEL_15:
    result = (_BYTE *)sub_16E7DE0(v2, 41);
    if ( !*(_BYTE *)(a1 + 81) )
      return result;
    goto LABEL_10;
  }
  qmemcpy(v9, "liveOnEntry", 11);
  result = (_BYTE *)(*(_QWORD *)(v2 + 24) + 11LL);
  *(_QWORD *)(v2 + 24) = result;
  if ( *(_QWORD *)(v2 + 16) <= (unsigned __int64)result )
    goto LABEL_15;
LABEL_9:
  *(_QWORD *)(v2 + 24) = result + 1;
  *result = 41;
  if ( !*(_BYTE *)(a1 + 81) )
    return result;
LABEL_10:
  v7 = *(_BYTE *)(a1 + 80);
  v8 = *(_BYTE **)(v2 + 24);
  if ( *(_BYTE **)(v2 + 16) == v8 )
  {
    v2 = sub_16E7EE0(v2, " ", 1);
  }
  else
  {
    *v8 = 32;
    ++*(_QWORD *)(v2 + 24);
  }
  return (_BYTE *)sub_134CED0(v2, v7);
}
