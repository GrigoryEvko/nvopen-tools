// Function: sub_16B2CD0
// Address: 0x16b2cd0
//
__int64 __fastcall sub_16B2CD0(void (__fastcall ***a1)(_QWORD), __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v4; // r14d
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // r12
  void *v10; // rdi
  unsigned __int64 v11; // r15
  const char *v12; // rsi
  __int64 v13; // rax
  size_t v14; // rdx
  __int64 v15; // rcx
  size_t v16; // r12
  const char *v17; // r15
  __int64 v18; // rax
  __int64 v19; // rdi
  __int64 v20; // rax
  int v21; // eax
  __int64 v23; // rax
  __int64 v24; // rdi
  __int64 v25; // rdi
  _BYTE *v26; // rax
  __int64 v27; // rax

  v4 = a3;
  v7 = sub_16E8C20(a1, a2, a3, a4);
  v8 = *(_QWORD *)(v7 + 24);
  v9 = v7;
  if ( (unsigned __int64)(*(_QWORD *)(v7 + 16) - v8) <= 2 )
  {
    v27 = sub_16E7EE0(v7, "  -", 3);
    v10 = *(void **)(v27 + 24);
    v9 = v27;
  }
  else
  {
    *(_BYTE *)(v8 + 2) = 45;
    *(_WORD *)v8 = 8224;
    v10 = (void *)(*(_QWORD *)(v7 + 24) + 3LL);
    *(_QWORD *)(v7 + 24) = v10;
  }
  v11 = *(_QWORD *)(a2 + 32);
  v12 = *(const char **)(a2 + 24);
  if ( *(_QWORD *)(v9 + 16) - (_QWORD)v10 < v11 )
  {
    sub_16E7EE0(v9, v12, *(_QWORD *)(a2 + 32));
  }
  else if ( v11 )
  {
    memcpy(v10, v12, *(_QWORD *)(a2 + 32));
    *(_QWORD *)(v9 + 24) += v11;
  }
  v13 = ((__int64 (__fastcall *)(void (__fastcall ***)(_QWORD), const char *))**a1)(a1, v12);
  v16 = v14;
  if ( v14 )
  {
    v17 = (const char *)v13;
    if ( (*(_BYTE *)(a2 + 13) & 4) != 0 )
    {
      v18 = sub_16E8C20(a1, v12, v14, v15);
      v19 = sub_1263B40(v18, " <");
      if ( *(_QWORD *)(a2 + 64) )
      {
        v17 = *(const char **)(a2 + 56);
        v16 = *(_QWORD *)(a2 + 64);
      }
      v20 = sub_1549FF0(v19, v17, v16);
      sub_1263B40(v20, ">...");
    }
    else
    {
      v23 = sub_16E8C20(a1, v12, v14, v15);
      v24 = sub_1263B40(v23, "=<");
      if ( *(_QWORD *)(a2 + 64) )
      {
        v17 = *(const char **)(a2 + 56);
        v16 = *(_QWORD *)(a2 + 64);
      }
      v25 = sub_1549FF0(v24, v17, v16);
      v26 = *(_BYTE **)(v25 + 24);
      if ( (unsigned __int64)v26 >= *(_QWORD *)(v25 + 16) )
      {
        sub_16E7DE0(v25, 62);
      }
      else
      {
        *(_QWORD *)(v25 + 24) = v26 + 1;
        *v26 = 62;
      }
    }
  }
  v21 = sub_16B2C10(a1, a2);
  return sub_16B2520(*(_OWORD *)(a2 + 40), v4, v21);
}
