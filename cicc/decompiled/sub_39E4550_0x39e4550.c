// Function: sub_39E4550
// Address: 0x39e4550
//
void __fastcall sub_39E4550(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  void *v4; // rdx
  unsigned __int64 v5; // r13
  __int64 v6; // rdi
  _BYTE *v7; // rax
  __int64 v8; // r14
  char *v9; // rsi
  size_t v10; // rdx
  void *v11; // rdi
  __int64 v12[5]; // [rsp+8h] [rbp-28h] BYREF

  if ( sub_38CF290(a2, v12) )
  {
    sub_38DCF20(a1, v12[0]);
  }
  else
  {
    v3 = *(_QWORD *)(a1 + 272);
    v4 = *(void **)(v3 + 24);
    if ( *(_QWORD *)(v3 + 16) - (_QWORD)v4 <= 9u )
    {
      sub_16E7EE0(v3, "\t.sleb128 ", 0xAu);
    }
    else
    {
      qmemcpy(v4, "\t.sleb128 ", 10);
      *(_QWORD *)(v3 + 24) += 10LL;
    }
    sub_38CDBE0(a2, *(_QWORD *)(a1 + 272), *(_QWORD *)(a1 + 280));
    v5 = *(unsigned int *)(a1 + 312);
    if ( *(_DWORD *)(a1 + 312) )
    {
      v8 = *(_QWORD *)(a1 + 272);
      v9 = *(char **)(a1 + 304);
      v10 = *(unsigned int *)(a1 + 312);
      v11 = *(void **)(v8 + 24);
      if ( v5 > *(_QWORD *)(v8 + 16) - (_QWORD)v11 )
      {
        sub_16E7EE0(*(_QWORD *)(a1 + 272), v9, v10);
      }
      else
      {
        memcpy(v11, v9, v10);
        *(_QWORD *)(v8 + 24) += v5;
      }
    }
    *(_DWORD *)(a1 + 312) = 0;
    if ( (*(_BYTE *)(a1 + 680) & 1) != 0 )
    {
      sub_39E0440(a1);
    }
    else
    {
      v6 = *(_QWORD *)(a1 + 272);
      v7 = *(_BYTE **)(v6 + 24);
      if ( (unsigned __int64)v7 >= *(_QWORD *)(v6 + 16) )
      {
        sub_16E7DE0(v6, 10);
      }
      else
      {
        *(_QWORD *)(v6 + 24) = v7 + 1;
        *v7 = 10;
      }
    }
  }
}
