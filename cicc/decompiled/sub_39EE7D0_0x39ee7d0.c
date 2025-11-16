// Function: sub_39EE7D0
// Address: 0x39ee7d0
//
void __fastcall sub_39EE7D0(__int64 a1, __int64 a2, __int64 a3)
{
  char *v6; // rsi
  __int64 v7; // rbx
  size_t v8; // rax
  __int64 v9; // rdx
  size_t v10; // r15
  void *v11; // rdi
  unsigned __int64 v12; // r13
  __int64 v13; // rdi
  _BYTE *v14; // rax
  __int64 v15; // r14
  char *v16; // rsi
  size_t v17; // rdx
  void *v18; // rdi
  __int64 v19; // rdi
  _BYTE *v20; // rax
  __int64 v21; // [rsp+8h] [rbp-48h]
  _QWORD v22[7]; // [rsp+18h] [rbp-38h] BYREF

  if ( !sub_38CF290(a2, v22) || v22[0] )
  {
    v6 = *(char **)(*(_QWORD *)(a1 + 280) + 176LL);
    v21 = *(_QWORD *)(a1 + 280);
    if ( v6 )
    {
      v7 = *(_QWORD *)(a1 + 272);
      v8 = strlen(v6);
      v9 = v21;
      v10 = v8;
      v11 = *(void **)(v7 + 24);
      if ( v8 > *(_QWORD *)(v7 + 16) - (_QWORD)v11 )
      {
        sub_16E7EE0(v7, v6, v8);
        v9 = *(_QWORD *)(a1 + 280);
        v7 = *(_QWORD *)(a1 + 272);
      }
      else if ( v8 )
      {
        memcpy(v11, v6, v8);
        *(_QWORD *)(v7 + 24) += v10;
        v9 = *(_QWORD *)(a1 + 280);
        v7 = *(_QWORD *)(a1 + 272);
      }
      sub_38CDBE0(a2, v7, v9);
      if ( a3 )
      {
        v19 = *(_QWORD *)(a1 + 272);
        v20 = *(_BYTE **)(v19 + 24);
        if ( (unsigned __int64)v20 >= *(_QWORD *)(v19 + 16) )
        {
          v19 = sub_16E7DE0(v19, 44);
        }
        else
        {
          *(_QWORD *)(v19 + 24) = v20 + 1;
          *v20 = 44;
        }
        sub_16E7AB0(v19, (int)a3);
      }
      v12 = *(unsigned int *)(a1 + 312);
      if ( *(_DWORD *)(a1 + 312) )
      {
        v15 = *(_QWORD *)(a1 + 272);
        v16 = *(char **)(a1 + 304);
        v17 = *(unsigned int *)(a1 + 312);
        v18 = *(void **)(v15 + 24);
        if ( v12 > *(_QWORD *)(v15 + 16) - (_QWORD)v18 )
        {
          sub_16E7EE0(*(_QWORD *)(a1 + 272), v16, v17);
        }
        else
        {
          memcpy(v18, v16, v17);
          *(_QWORD *)(v15 + 24) += v12;
        }
      }
      *(_DWORD *)(a1 + 312) = 0;
      if ( (*(_BYTE *)(a1 + 680) & 1) != 0 )
      {
        sub_39E0440(a1);
      }
      else
      {
        v13 = *(_QWORD *)(a1 + 272);
        v14 = *(_BYTE **)(v13 + 24);
        if ( (unsigned __int64)v14 >= *(_QWORD *)(v13 + 16) )
        {
          sub_16E7DE0(v13, 10);
        }
        else
        {
          *(_QWORD *)(v13 + 24) = v14 + 1;
          *v14 = 10;
        }
      }
    }
    else
    {
      nullsub_1954();
    }
  }
}
