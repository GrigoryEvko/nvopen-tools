// Function: sub_BBB850
// Address: 0xbbb850
//
_BYTE *__fastcall sub_BBB850(__int64 a1, __int64 a2)
{
  void *v3; // rdx
  __int64 *v4; // rdi
  __int64 v5; // rax
  __int64 v6; // rdx
  _BYTE *result; // rax
  __int64 v8; // rdx
  _QWORD v9[3]; // [rsp+8h] [rbp-18h] BYREF

  v3 = *(void **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v3 <= 0xDu )
  {
    sub_CB6200(a2, "Running pass \"", 14);
    v4 = *(__int64 **)(a1 + 32);
    if ( v4 )
      goto LABEL_3;
LABEL_9:
    v8 = *(_QWORD *)(a2 + 32);
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v8) <= 6 )
    {
      sub_CB6200(a2, "unknown", 7);
      v6 = *(_QWORD *)(a2 + 32);
    }
    else
    {
      *(_DWORD *)v8 = 1852534389;
      *(_WORD *)(v8 + 4) = 30575;
      *(_BYTE *)(v8 + 6) = 110;
      v6 = *(_QWORD *)(a2 + 32) + 7LL;
      *(_QWORD *)(a2 + 32) = v6;
    }
    goto LABEL_4;
  }
  qmemcpy(v3, "Running pass \"", 14);
  *(_QWORD *)(a2 + 32) += 14LL;
  v4 = *(__int64 **)(a1 + 32);
  if ( !v4 )
    goto LABEL_9;
LABEL_3:
  v5 = *v4;
  v9[0] = a1;
  (*(void (__fastcall **)(__int64 *, __int64, __int64 (__fastcall *)(__int64, __int64), _QWORD *))(v5 + 24))(
    v4,
    a2,
    sub_BBACB0,
    v9);
  v6 = *(_QWORD *)(a2 + 32);
LABEL_4:
  if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v6) <= 4 )
  {
    sub_CB6200(a2, "\" on ", 5);
  }
  else
  {
    *(_DWORD *)v6 = 1852776482;
    *(_BYTE *)(v6 + 4) = 32;
    *(_QWORD *)(a2 + 32) += 5LL;
  }
  sub_BBB750(a2, *(_QWORD *)(a1 + 24));
  result = *(_BYTE **)(a2 + 32);
  if ( *(_BYTE **)(a2 + 24) == result )
    return (_BYTE *)sub_CB6200(a2, "\n", 1);
  *result = 10;
  ++*(_QWORD *)(a2 + 32);
  return result;
}
