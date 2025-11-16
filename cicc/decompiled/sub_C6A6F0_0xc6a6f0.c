// Function: sub_C6A6F0
// Address: 0xc6a6f0
//
void __fastcall sub_C6A6F0(__int64 a1)
{
  const char *v1; // r13
  __int64 v3; // r14
  size_t v4; // rax
  __int64 v5; // rcx
  size_t v6; // rdx
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rbx
  __int64 v9; // r15
  size_t v10; // rdx
  const void *v11; // rsi
  _BYTE *v12; // rdi
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // rdx
  unsigned __int64 v15; // rbx
  __int64 v16; // rax
  unsigned __int64 v17; // rdx
  __int64 v18; // r13
  unsigned __int64 v19; // r14
  const void *v20; // rsi
  void *v21; // rdi
  const char *v22; // r14
  size_t v23; // rax
  __int64 v24; // rcx
  size_t v25; // rdx
  unsigned __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rax
  unsigned int v29; // eax
  __int64 v30; // rsi
  __int64 v31; // rdi
  _BYTE *v32; // rax
  unsigned int v33; // eax
  __int64 v34; // rsi
  size_t v35; // [rsp-40h] [rbp-40h]

  if ( *(_QWORD *)(a1 + 152) )
  {
    v1 = "/* ";
    v3 = *(_QWORD *)(a1 + 160);
    if ( !*(_DWORD *)(a1 + 168) )
      v1 = "/*";
    v4 = strlen(v1);
    v5 = *(_QWORD *)(v3 + 32);
    v6 = v4;
    if ( v4 <= *(_QWORD *)(v3 + 24) - v5 )
    {
      if ( (_DWORD)v4 )
      {
        v29 = 0;
        do
        {
          v30 = v29++;
          *(_BYTE *)(v5 + v30) = v1[v30];
        }
        while ( v29 < (unsigned int)v6 );
      }
      *(_QWORD *)(v3 + 32) += v6;
    }
    else
    {
      sub_CB6200(v3, v1, v4);
    }
    if ( *(_QWORD *)(a1 + 152) )
    {
      while ( 1 )
      {
        v7 = sub_C931B0(a1 + 144, "*/", 2, 0);
        v8 = v7;
        if ( v7 == -1 )
          break;
        v9 = *(_QWORD *)(a1 + 160);
        v10 = v7;
        if ( *(_QWORD *)(a1 + 152) <= v7 )
          v10 = *(_QWORD *)(a1 + 152);
        v11 = *(const void **)(a1 + 144);
        v12 = *(_BYTE **)(v9 + 32);
        v13 = *(_QWORD *)(v9 + 24) - (_QWORD)v12;
        if ( v13 < v10 )
        {
          v27 = sub_CB6200(*(_QWORD *)(a1 + 160), v11, v10);
          v12 = *(_BYTE **)(v27 + 32);
          v9 = v27;
          v13 = *(_QWORD *)(v27 + 24) - (_QWORD)v12;
        }
        else if ( v10 )
        {
          v35 = v10;
          memcpy(v12, v11, v10);
          v28 = *(_QWORD *)(v9 + 24);
          v12 = (_BYTE *)(v35 + *(_QWORD *)(v9 + 32));
          *(_QWORD *)(v9 + 32) = v12;
          v13 = v28 - (_QWORD)v12;
        }
        if ( v13 <= 2 )
        {
          sub_CB6200(v9, "* /", 3);
        }
        else
        {
          v12[2] = 47;
          *(_WORD *)v12 = 8234;
          *(_QWORD *)(v9 + 32) += 3LL;
        }
        v14 = *(_QWORD *)(a1 + 152);
        v15 = v8 + 2;
        v16 = *(_QWORD *)(a1 + 144);
        if ( v15 > v14 )
        {
          v18 = *(_QWORD *)(a1 + 160);
          *(_QWORD *)(a1 + 152) = 0;
          *(_QWORD *)(a1 + 144) = v14 + v16;
          goto LABEL_22;
        }
        v17 = v14 - v15;
        *(_QWORD *)(a1 + 144) = v16 + v15;
        *(_QWORD *)(a1 + 152) = v17;
        if ( !v17 )
          goto LABEL_17;
      }
      v18 = *(_QWORD *)(a1 + 160);
      v19 = *(_QWORD *)(a1 + 152);
      v20 = *(const void **)(a1 + 144);
      v21 = *(void **)(v18 + 32);
      if ( *(_QWORD *)(v18 + 24) - (_QWORD)v21 < v19 )
      {
        sub_CB6200(*(_QWORD *)(a1 + 160), v20, *(_QWORD *)(a1 + 152));
        v18 = *(_QWORD *)(a1 + 160);
      }
      else if ( v19 )
      {
        memcpy(v21, v20, *(_QWORD *)(a1 + 152));
        *(_QWORD *)(v18 + 32) += v19;
        v18 = *(_QWORD *)(a1 + 160);
      }
      *(_QWORD *)(a1 + 152) = 0;
      *(_QWORD *)(a1 + 144) = byte_3F871B3;
    }
    else
    {
LABEL_17:
      v18 = *(_QWORD *)(a1 + 160);
    }
LABEL_22:
    v22 = (const char *)&unk_3F66CB0;
    if ( !*(_DWORD *)(a1 + 168) )
      v22 = "*/";
    v23 = strlen(v22);
    v24 = *(_QWORD *)(v18 + 32);
    v25 = v23;
    if ( v23 <= *(_QWORD *)(v18 + 24) - v24 )
    {
      if ( (_DWORD)v23 )
      {
        v33 = 0;
        do
        {
          v34 = v33++;
          *(_BYTE *)(v24 + v34) = v22[v34];
        }
        while ( v33 < (unsigned int)v25 );
      }
      *(_QWORD *)(v18 + 32) += v25;
    }
    else
    {
      sub_CB6200(v18, v22, v23);
    }
    v26 = *(unsigned int *)(a1 + 8);
    if ( v26 <= 1 || *(_DWORD *)(*(_QWORD *)a1 + 8 * v26 - 8) )
    {
      sub_C6A6A0(a1);
    }
    else if ( *(_DWORD *)(a1 + 168) )
    {
      v31 = *(_QWORD *)(a1 + 160);
      v32 = *(_BYTE **)(v31 + 32);
      if ( (unsigned __int64)v32 >= *(_QWORD *)(v31 + 24) )
      {
        sub_CB5D20(v31, 32);
      }
      else
      {
        *(_QWORD *)(v31 + 32) = v32 + 1;
        *v32 = 32;
      }
    }
  }
}
