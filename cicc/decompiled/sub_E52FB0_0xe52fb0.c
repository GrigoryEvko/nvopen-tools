// Function: sub_E52FB0
// Address: 0xe52fb0
//
void __fastcall sub_E52FB0(__int64 a1, char *a2, __int64 a3)
{
  __int64 v5; // rdi
  __int64 v6; // rax
  const char *v7; // rsi
  __int64 v8; // r13
  const char *v9; // rsi
  char *v10; // r15
  char *v11; // rbx
  unsigned __int8 v12; // r14
  __int64 v13; // r13
  size_t v14; // rax
  void *v15; // rdi
  unsigned __int64 v16; // r14
  bool v17; // zf
  __int64 v18; // rdi
  _BYTE *v19; // rax
  __int64 v20; // r8
  unsigned __int8 *v21; // rsi
  void *v22; // rdi
  char *v23; // r15
  char *v24; // rcx
  char *v25; // rbx
  __int64 v26; // rdi
  __int64 v27; // r13
  int v28; // eax
  unsigned __int8 v29; // r14
  unsigned __int8 v30; // r14
  unsigned __int8 v31; // bl
  _BYTE *v32; // rax
  char *v33; // rdx
  char v34; // al
  _BYTE *v35; // rdx
  char *v36; // rax
  char v37; // bl
  size_t v38; // [rsp-58h] [rbp-58h]
  __int64 v39; // [rsp-58h] [rbp-58h]
  char *v40; // [rsp-50h] [rbp-50h]
  char *v42; // [rsp-50h] [rbp-50h]
  char v44; // [rsp-3Ah] [rbp-3Ah] BYREF
  unsigned __int8 v45; // [rsp-39h] [rbp-39h]

  if ( !a3 )
    return;
  if ( a3 != 1 )
  {
    v6 = *(_QWORD *)(a1 + 312);
    if ( !*(_BYTE *)(v6 + 21) )
    {
      v7 = *(const char **)(v6 + 208);
      v8 = a3;
      if ( v7 && !a2[a3 - 1] )
      {
        v8 = a3 - 1;
        sub_904010(*(_QWORD *)(a1 + 304), v7);
      }
      else
      {
        v9 = *(const char **)(v6 + 200);
        if ( !v9 )
          goto LABEL_3;
        sub_904010(*(_QWORD *)(a1 + 304), v9);
      }
LABEL_10:
      sub_E51560(a1, a2, v8, *(_QWORD *)(a1 + 304));
LABEL_11:
      sub_E4D880(a1);
      return;
    }
    v8 = a3 - 1;
    v23 = a2;
    v24 = a2;
    v25 = &a2[a3 - 1];
    do
    {
      if ( (unsigned __int8)(*v24 - 32) > 0x5Eu )
      {
        v26 = *(_QWORD *)(a1 + 304);
        goto LABEL_35;
      }
      ++v24;
    }
    while ( v25 != v24 );
    v26 = *(_QWORD *)(a1 + 304);
    if ( (unsigned __int8)(*v25 - 32) <= 0x5Eu )
    {
      sub_904010(v26, "\t.byte\t");
      v8 = a3;
      goto LABEL_10;
    }
    if ( !*v25 )
    {
      sub_904010(v26, "\t.string\t");
      goto LABEL_10;
    }
LABEL_35:
    sub_904010(v26, "\t.byte\t");
    v27 = *(_QWORD *)(a1 + 304);
    v28 = *(_DWORD *)(*(_QWORD *)(a1 + 312) + 216LL);
    if ( v28 )
    {
      if ( v28 != 1 )
        BUG();
      do
      {
        v29 = *v23;
        if ( (unsigned __int8)(*v23 - 32) > 0x5Eu )
        {
          sub_A51310(v27, 0x30u);
          sub_A51310(v27, (v29 >> 6) + 48);
          sub_A51310(v27, ((v29 >> 3) & 7) + 48);
          sub_A51310(v27, (v29 & 7) + 48);
        }
        else
        {
          v44 = 39;
          v45 = v29;
          sub_A51340(v27, &v44, 2u);
        }
        ++v23;
        sub_A51310(v27, 0x2Cu);
      }
      while ( v25 != v23 );
      v31 = a2[a3 - 1];
      if ( (unsigned __int8)(v31 - 32) <= 0x5Eu )
      {
        v44 = 39;
        v45 = v31;
        sub_A51340(v27, &v44, 2u);
        goto LABEL_11;
      }
      v32 = *(_BYTE **)(v27 + 32);
      if ( (unsigned __int64)v32 >= *(_QWORD *)(v27 + 24) )
        goto LABEL_59;
    }
    else
    {
      v42 = &a2[a3];
      do
      {
        v30 = *v23++;
        sub_A51310(v27, 0x30u);
        sub_A51310(v27, (v30 >> 6) + 48);
        sub_A51310(v27, ((v30 >> 3) & 7) + 48);
        sub_A51310(v27, (v30 & 7) + 48);
        sub_A51310(v27, 0x2Cu);
      }
      while ( v25 != v23 );
      v31 = *(v42 - 1);
      v32 = *(_BYTE **)(v27 + 32);
      if ( (unsigned __int64)v32 >= *(_QWORD *)(v27 + 24) )
      {
LABEL_59:
        sub_CB5D20(v27, 48);
        goto LABEL_49;
      }
    }
    *(_QWORD *)(v27 + 32) = v32 + 1;
    *v32 = 48;
LABEL_49:
    v33 = *(char **)(v27 + 32);
    v34 = (v31 >> 6) + 48;
    if ( (unsigned __int64)v33 >= *(_QWORD *)(v27 + 24) )
    {
      sub_CB5D20(v27, v34);
    }
    else
    {
      *(_QWORD *)(v27 + 32) = v33 + 1;
      *v33 = v34;
    }
    v35 = *(_BYTE **)(v27 + 32);
    if ( (unsigned __int64)v35 >= *(_QWORD *)(v27 + 24) )
    {
      sub_CB5D20(v27, ((v31 >> 3) & 7) + 48);
    }
    else
    {
      *(_QWORD *)(v27 + 32) = v35 + 1;
      *v35 = ((v31 >> 3) & 7) + 48;
    }
    v36 = *(char **)(v27 + 32);
    v37 = (v31 & 7) + 48;
    if ( (unsigned __int64)v36 >= *(_QWORD *)(v27 + 24) )
    {
      sub_CB5D20(v27, v37);
    }
    else
    {
      *(_QWORD *)(v27 + 32) = v36 + 1;
      *v36 = v37;
    }
    goto LABEL_11;
  }
LABEL_3:
  v5 = *(_QWORD *)(a1 + 16);
  if ( v5 )
  {
    (*(void (__fastcall **)(__int64, char *))(*(_QWORD *)v5 + 64LL))(v5, a2);
    return;
  }
  v10 = *(char **)(*(_QWORD *)(a1 + 312) + 224LL);
  v40 = &a2[a3];
  if ( &a2[a3] != a2 )
  {
    v11 = a2;
    do
    {
      while ( 1 )
      {
        v12 = *v11;
        v13 = *(_QWORD *)(a1 + 304);
        if ( v10 )
        {
          v14 = strlen(v10);
          v15 = *(void **)(v13 + 32);
          if ( v14 > *(_QWORD *)(v13 + 24) - (_QWORD)v15 )
          {
            v13 = sub_CB6200(v13, (unsigned __int8 *)v10, v14);
          }
          else if ( v14 )
          {
            v38 = v14;
            memcpy(v15, v10, v14);
            *(_QWORD *)(v13 + 32) += v38;
          }
        }
        sub_CB59D0(v13, v12);
        v16 = *(_QWORD *)(a1 + 344);
        if ( v16 )
        {
          v20 = *(_QWORD *)(a1 + 304);
          v21 = *(unsigned __int8 **)(a1 + 336);
          v22 = *(void **)(v20 + 32);
          if ( v16 > *(_QWORD *)(v20 + 24) - (_QWORD)v22 )
          {
            sub_CB6200(*(_QWORD *)(a1 + 304), v21, *(_QWORD *)(a1 + 344));
          }
          else
          {
            v39 = *(_QWORD *)(a1 + 304);
            memcpy(v22, v21, *(_QWORD *)(a1 + 344));
            *(_QWORD *)(v39 + 32) += v16;
          }
        }
        v17 = *(_BYTE *)(a1 + 745) == 0;
        *(_QWORD *)(a1 + 344) = 0;
        if ( v17 )
          break;
        sub_E4D630((__int64 *)a1);
LABEL_15:
        if ( v40 == ++v11 )
          return;
      }
      v18 = *(_QWORD *)(a1 + 304);
      v19 = *(_BYTE **)(v18 + 32);
      if ( (unsigned __int64)v19 >= *(_QWORD *)(v18 + 24) )
      {
        sub_CB5D20(v18, 10);
        goto LABEL_15;
      }
      ++v11;
      *(_QWORD *)(v18 + 32) = v19 + 1;
      *v19 = 10;
    }
    while ( v40 != v11 );
  }
}
