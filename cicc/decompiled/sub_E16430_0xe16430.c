// Function: sub_E16430
// Address: 0xe16430
//
char *__fastcall sub_E16430(__int64 a1, __int64 a2)
{
  _QWORD *v3; // r13
  __int64 v5; // rdi
  char *result; // rax
  _BYTE *v7; // r12
  __int64 v8; // rsi
  unsigned __int64 v9; // rax
  __int64 v10; // rdx
  unsigned __int64 v11; // rsi
  unsigned __int64 v12; // rax

  v3 = (_QWORD *)(a1 + 24);
  v5 = *(_QWORD *)(a1 + 16);
  if ( v5 )
  {
    result = (char *)(*(__int64 (__fastcall **)(__int64, __int64, _QWORD *))(*(_QWORD *)v5 + 48LL))(v5, a2, v3);
    if ( (_BYTE)result )
      return result;
    v7 = *(_BYTE **)(a1 + 16);
    (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v7 + 32LL))(v7, a2);
    if ( (v7[9] & 0xC0) != 0x40 )
      (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v7 + 40LL))(v7, a2);
  }
  sub_E14360(a2, 123);
  sub_E161C0(v3, (char **)a2);
  v8 = *(_QWORD *)(a2 + 8);
  v9 = *(_QWORD *)(a2 + 16);
  v10 = v8 + 1;
  if ( v8 + 1 > v9 )
  {
    v11 = v8 + 993;
    v12 = 2 * v9;
    if ( v11 > v12 )
      *(_QWORD *)(a2 + 16) = v11;
    else
      *(_QWORD *)(a2 + 16) = v12;
    result = (char *)realloc(*(void **)a2);
    *(_QWORD *)a2 = result;
    if ( !result )
      abort();
    v8 = *(_QWORD *)(a2 + 8);
    v10 = v8 + 1;
  }
  else
  {
    result = *(char **)a2;
  }
  *(_QWORD *)(a2 + 8) = v10;
  result[v8] = 125;
  return result;
}
