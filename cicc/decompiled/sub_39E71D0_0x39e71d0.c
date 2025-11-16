// Function: sub_39E71D0
// Address: 0x39e71d0
//
void __fastcall sub_39E71D0(__int64 a1, unsigned __int8 *a2, __int64 a3)
{
  int v4; // r13d
  __int64 v6; // rax
  const char *v7; // rsi
  __int64 v8; // r14
  __int64 v9; // rbx
  char *v10; // rsi
  size_t v11; // rax
  void *v12; // rdi
  size_t v13; // r15
  unsigned __int64 v14; // r13
  __int64 v15; // rdi
  _BYTE *v16; // rax
  __int64 v17; // rdi
  __int64 v18; // r14
  char *v19; // rsi
  size_t v20; // rdx
  void *v21; // rdi
  char *v22; // r15
  unsigned __int8 *v23; // r13
  unsigned __int8 v24; // r14
  __int64 v25; // rbx
  size_t v26; // rax
  void *v27; // rdi
  unsigned __int8 *v28; // [rsp-48h] [rbp-48h]
  size_t v29; // [rsp-48h] [rbp-48h]
  unsigned __int8 *v30; // [rsp-40h] [rbp-40h]
  unsigned __int8 *v31; // [rsp-40h] [rbp-40h]

  if ( !a3 )
    return;
  v4 = a3;
  if ( a3 != 1 )
  {
    v6 = *(_QWORD *)(a1 + 280);
    v7 = *(const char **)(v6 + 192);
    if ( v7 )
    {
      v8 = *(_QWORD *)(a1 + 272);
      v9 = v8;
      if ( a2[a3 - 1] )
      {
        v10 = *(char **)(v6 + 184);
        if ( v10 )
          goto LABEL_6;
      }
      else
      {
        v30 = a2;
        v4 = a3 - 1;
        sub_1263B40(*(_QWORD *)(a1 + 272), v7);
        v8 = *(_QWORD *)(a1 + 272);
        a2 = v30;
      }
LABEL_9:
      sub_39E0070(a2, v4, v8);
      v14 = *(unsigned int *)(a1 + 312);
      if ( *(_DWORD *)(a1 + 312) )
      {
        v18 = *(_QWORD *)(a1 + 272);
        v19 = *(char **)(a1 + 304);
        v20 = *(unsigned int *)(a1 + 312);
        v21 = *(void **)(v18 + 24);
        if ( v14 > *(_QWORD *)(v18 + 16) - (_QWORD)v21 )
        {
          sub_16E7EE0(*(_QWORD *)(a1 + 272), v19, v20);
        }
        else
        {
          memcpy(v21, v19, v20);
          *(_QWORD *)(v18 + 24) += v14;
        }
      }
      *(_DWORD *)(a1 + 312) = 0;
      if ( (*(_BYTE *)(a1 + 680) & 1) != 0 )
      {
        sub_39E0440(a1);
      }
      else
      {
        v15 = *(_QWORD *)(a1 + 272);
        v16 = *(_BYTE **)(v15 + 24);
        if ( (unsigned __int64)v16 >= *(_QWORD *)(v15 + 16) )
        {
          sub_16E7DE0(v15, 10);
        }
        else
        {
          *(_QWORD *)(v15 + 24) = v16 + 1;
          *v16 = 10;
        }
      }
      return;
    }
    v10 = *(char **)(v6 + 184);
    if ( v10 )
    {
      v9 = *(_QWORD *)(a1 + 272);
      v8 = v9;
LABEL_6:
      v28 = a2;
      v11 = strlen(v10);
      v12 = *(void **)(v9 + 24);
      v13 = v11;
      a2 = v28;
      if ( v11 > *(_QWORD *)(v9 + 16) - (_QWORD)v12 )
      {
        sub_16E7EE0(v8, v10, v11);
        v8 = *(_QWORD *)(a1 + 272);
        a2 = v28;
      }
      else if ( v11 )
      {
        memcpy(v12, v10, v11);
        *(_QWORD *)(v9 + 24) += v13;
        a2 = v28;
        v8 = *(_QWORD *)(a1 + 272);
      }
      goto LABEL_9;
    }
  }
  v17 = *(_QWORD *)(a1 + 16);
  if ( v17 )
  {
    (*(void (__fastcall **)(__int64, unsigned __int8 *))(*(_QWORD *)v17 + 64LL))(v17, a2);
  }
  else
  {
    v22 = *(char **)(*(_QWORD *)(a1 + 280) + 200LL);
    v31 = &a2[a3];
    if ( a2 != &a2[a3] )
    {
      v23 = a2;
      do
      {
        v24 = *v23;
        v25 = *(_QWORD *)(a1 + 272);
        if ( v22 )
        {
          v26 = strlen(v22);
          v27 = *(void **)(v25 + 24);
          if ( v26 <= *(_QWORD *)(v25 + 16) - (_QWORD)v27 )
          {
            if ( v26 )
            {
              v29 = v26;
              memcpy(v27, v22, v26);
              *(_QWORD *)(v25 + 24) += v29;
            }
          }
          else
          {
            v25 = sub_16E7EE0(v25, v22, v26);
          }
        }
        ++v23;
        sub_16E7A90(v25, v24);
        sub_39E06C0(a1);
      }
      while ( v31 != v23 );
    }
  }
}
