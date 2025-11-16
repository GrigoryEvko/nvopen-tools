// Function: sub_183B9A0
// Address: 0x183b9a0
//
void *__fastcall sub_183B9A0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  unsigned __int64 v5; // rdi
  const char *v6; // rax
  size_t v7; // rdx
  void *v8; // rdi
  char *v9; // rsi
  void *result; // rax
  __int64 v11; // rdx
  __int64 v12; // rdx
  __int64 v13; // rdx
  size_t v14; // [rsp+8h] [rbp-18h]

  v4 = (a2 >> 1) & 3;
  if ( ((a2 >> 1) & 3) != 0 )
  {
    if ( v4 == 2 )
    {
      v13 = *(_QWORD *)(a3 + 24);
      if ( (unsigned __int64)(*(_QWORD *)(a3 + 16) - v13) <= 5 )
      {
        sub_16E7EE0(a3, "<mem> ", 6u);
      }
      else
      {
        *(_DWORD *)v13 = 1835363644;
        *(_WORD *)(v13 + 4) = 8254;
        *(_QWORD *)(a3 + 24) += 6LL;
      }
    }
    else if ( (_DWORD)v4 == 1 )
    {
      v12 = *(_QWORD *)(a3 + 24);
      if ( (unsigned __int64)(*(_QWORD *)(a3 + 16) - v12) <= 5 )
      {
        sub_16E7EE0(a3, "<ret> ", 6u);
      }
      else
      {
        *(_DWORD *)v12 = 1952805436;
        *(_WORD *)(v12 + 4) = 8254;
        *(_QWORD *)(a3 + 24) += 6LL;
      }
    }
  }
  else
  {
    v11 = *(_QWORD *)(a3 + 24);
    if ( (unsigned __int64)(*(_QWORD *)(a3 + 16) - v11) <= 5 )
    {
      sub_16E7EE0(a3, "<reg> ", 6u);
    }
    else
    {
      *(_DWORD *)v11 = 1734701628;
      *(_WORD *)(v11 + 4) = 8254;
      *(_QWORD *)(a3 + 24) += 6LL;
    }
  }
  v5 = a2 & 0xFFFFFFFFFFFFFFF8LL;
  if ( *(_BYTE *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 16) )
    return (void *)sub_155C2B0(v5, a3, 0);
  v6 = sub_1649960(v5);
  v8 = *(void **)(a3 + 24);
  v9 = (char *)v6;
  result = (void *)(*(_QWORD *)(a3 + 16) - (_QWORD)v8);
  if ( (unsigned __int64)result < v7 )
    return (void *)sub_16E7EE0(a3, v9, v7);
  if ( v7 )
  {
    v14 = v7;
    result = memcpy(v8, v9, v7);
    *(_QWORD *)(a3 + 24) += v14;
  }
  return result;
}
