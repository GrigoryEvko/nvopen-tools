// Function: sub_2977B90
// Address: 0x2977b90
//
unsigned __int64 __fastcall sub_2977B90(
        _BYTE *a1,
        __int64 a2,
        __int64 (__fastcall *a3)(__int64, char *, __int64),
        __int64 a4)
{
  __int64 v6; // rax
  size_t v7; // rdx
  void *v8; // rdi
  unsigned __int8 *v9; // rsi
  unsigned __int64 result; // rax
  void *v11; // rdx
  size_t v12; // [rsp+8h] [rbp-18h]

  v6 = a3(a4, "SinkingPass]", 11);
  v8 = *(void **)(a2 + 32);
  v9 = (unsigned __int8 *)v6;
  result = *(_QWORD *)(a2 + 24) - (_QWORD)v8;
  if ( result < v7 )
  {
    result = sub_CB6200(a2, v9, v7);
  }
  else if ( v7 )
  {
    v12 = v7;
    result = (unsigned __int64)memcpy(v8, v9, v7);
    *(_QWORD *)(a2 + 32) += v12;
    if ( !*a1 )
      return result;
    goto LABEL_6;
  }
  if ( !*a1 )
    return result;
LABEL_6:
  v11 = *(void **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v11 <= 9u )
    return sub_CB6200(a2, (unsigned __int8 *)"<rp-aware>", 0xAu);
  qmemcpy(v11, "<rp-aware>", 10);
  *(_QWORD *)(a2 + 32) += 10LL;
  return 15973;
}
