// Function: sub_39EB670
// Address: 0x39eb670
//
_BYTE *__fastcall sub_39EB670(__int64 a1, unsigned __int64 a2)
{
  int v3; // r8d
  int v4; // r9d
  char v5; // al
  unsigned __int64 v6; // rsi
  _BYTE *v7; // rdx
  unsigned __int64 v8; // r13
  _BYTE *result; // rax
  __int64 v10; // rdi
  __int64 v11; // r14
  char *v12; // rsi
  size_t v13; // rdx
  void *v14; // rdi
  _QWORD v15[6]; // [rsp+0h] [rbp-30h] BYREF

  sub_38DEEF0(a1, a2);
  v15[0] = 46;
  v5 = a2 & 0x7F;
  v15[1] = 0;
  if ( a2 >> 7 )
  {
    v6 = a2 >> 7;
    v7 = (char *)v15 + 1;
    do
    {
      *v7++ = v5 | 0x80;
      v5 = v6 & 0x7F;
      v6 >>= 7;
    }
    while ( v6 );
  }
  else
  {
    v7 = (char *)v15 + 1;
  }
  *v7 = v5;
  sub_39DFF00(
    *(_QWORD *)(a1 + 272),
    (char *)v15,
    (unsigned int)v7 - ((unsigned int)v15 + 1) + 2,
    (__int64)v15 + 1,
    v3,
    v4);
  v8 = *(unsigned int *)(a1 + 312);
  if ( *(_DWORD *)(a1 + 312) )
  {
    v11 = *(_QWORD *)(a1 + 272);
    v12 = *(char **)(a1 + 304);
    v13 = *(unsigned int *)(a1 + 312);
    v14 = *(void **)(v11 + 24);
    if ( v8 > *(_QWORD *)(v11 + 16) - (_QWORD)v14 )
    {
      sub_16E7EE0(*(_QWORD *)(a1 + 272), v12, v13);
    }
    else
    {
      memcpy(v14, v12, v13);
      *(_QWORD *)(v11 + 24) += v8;
    }
  }
  *(_DWORD *)(a1 + 312) = 0;
  if ( (*(_BYTE *)(a1 + 680) & 1) != 0 )
    return sub_39E0440(a1);
  v10 = *(_QWORD *)(a1 + 272);
  result = *(_BYTE **)(v10 + 24);
  if ( (unsigned __int64)result >= *(_QWORD *)(v10 + 16) )
    return (_BYTE *)sub_16E7DE0(v10, 10);
  *(_QWORD *)(v10 + 24) = result + 1;
  *result = 10;
  return result;
}
