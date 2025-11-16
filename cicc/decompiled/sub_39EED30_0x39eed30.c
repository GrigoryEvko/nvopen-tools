// Function: sub_39EED30
// Address: 0x39eed30
//
_BYTE *__fastcall sub_39EED30(__int64 a1)
{
  _BYTE *result; // rax
  _BYTE *v3; // r13
  __int64 v4; // rax
  __int64 v5; // r15
  char *v6; // r14
  size_t v7; // rax
  void *v8; // rdi
  size_t v9; // r13
  unsigned __int64 v10; // r13
  __int64 v11; // rdi
  __int64 v12; // r14
  char *v13; // rsi
  size_t v14; // rdx
  void *v15; // rdi

  result = *(_BYTE **)(a1 + 8);
  if ( result[1041] )
  {
    sub_38C9E40((__int64 *)a1);
    result = *(_BYTE **)(a1 + 8);
  }
  if ( *((_QWORD *)result + 127) )
  {
    v3 = *(_BYTE **)(*((_QWORD *)result + 125) + 40LL);
    if ( v3 )
    {
      sub_38DC390(a1, *(_QWORD *)(*((_QWORD *)result + 4) + 88LL), 0);
      sub_38DC4E0(a1, (__int64)v3, 0);
      sub_38E2490(v3, *(_QWORD *)(a1 + 272), *(_BYTE **)(a1 + 280));
      v4 = *(_QWORD *)(a1 + 280);
      v5 = *(_QWORD *)(a1 + 272);
      v6 = *(char **)(v4 + 64);
      if ( v6 )
      {
        v7 = strlen(*(const char **)(v4 + 64));
        v8 = *(void **)(v5 + 24);
        v9 = v7;
        if ( v7 > *(_QWORD *)(v5 + 16) - (_QWORD)v8 )
        {
          sub_16E7EE0(v5, v6, v7);
        }
        else if ( v7 )
        {
          memcpy(v8, v6, v7);
          *(_QWORD *)(v5 + 24) += v9;
        }
      }
      v10 = *(unsigned int *)(a1 + 312);
      if ( *(_DWORD *)(a1 + 312) )
      {
        v12 = *(_QWORD *)(a1 + 272);
        v13 = *(char **)(a1 + 304);
        v14 = *(unsigned int *)(a1 + 312);
        v15 = *(void **)(v12 + 24);
        if ( v10 > *(_QWORD *)(v12 + 16) - (_QWORD)v15 )
        {
          sub_16E7EE0(*(_QWORD *)(a1 + 272), v13, v14);
        }
        else
        {
          memcpy(v15, v13, v14);
          *(_QWORD *)(v12 + 24) += v10;
        }
      }
      *(_DWORD *)(a1 + 312) = 0;
      if ( (*(_BYTE *)(a1 + 680) & 1) != 0 )
      {
        return sub_39E0440(a1);
      }
      else
      {
        v11 = *(_QWORD *)(a1 + 272);
        result = *(_BYTE **)(v11 + 24);
        if ( (unsigned __int64)result >= *(_QWORD *)(v11 + 16) )
        {
          return (_BYTE *)sub_16E7DE0(v11, 10);
        }
        else
        {
          *(_QWORD *)(v11 + 24) = result + 1;
          *result = 10;
        }
      }
    }
  }
  return result;
}
