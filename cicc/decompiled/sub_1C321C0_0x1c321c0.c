// Function: sub_1C321C0
// Address: 0x1c321c0
//
__int64 __fastcall sub_1C321C0(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v5; // rdi
  _WORD *v6; // rdx
  __int64 v7; // rsi
  __int64 v8; // r12
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rdi
  void *v13; // rax
  size_t v14; // rdx
  _BYTE *v15; // rdi
  size_t v16; // r15
  unsigned int v17; // eax
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rdi
  void *v21; // rdx
  __int64 v22; // rdi
  __int64 v23; // rdx
  void *v25; // rdx
  const char *v26; // rax
  size_t v27; // rdx
  void *v28; // rdi
  char *v29; // rsi
  unsigned __int64 v30; // rax
  const char *v31; // rax
  size_t v32; // rdx
  _BYTE *v33; // rdi
  char *v34; // rsi
  unsigned __int64 v35; // rax
  __int64 v36; // rax
  _BYTE *v37; // rdx
  __int64 v38; // rax
  void *v39; // rdx
  __int64 v40; // rax
  __int64 v41; // rax
  size_t v42; // [rsp+8h] [rbp-48h]
  size_t v43; // [rsp+8h] [rbp-48h]
  __int64 v44[7]; // [rsp+18h] [rbp-38h] BYREF

  sub_1C31A90(a3, *(_QWORD *)(a1 + 24));
  v5 = *(_QWORD *)(a1 + 24);
  v6 = *(_WORD **)(v5 + 24);
  if ( *(_QWORD *)(v5 + 16) - (_QWORD)v6 <= 1u )
  {
    sub_16E7EE0(v5, ": ", 2u);
  }
  else
  {
    *v6 = 8250;
    *(_QWORD *)(v5 + 24) += 2LL;
  }
  v7 = *(_QWORD *)(a2 + 48);
  v8 = *(_QWORD *)(a1 + 24);
  v44[0] = v7;
  if ( v7 )
  {
    sub_1623A60((__int64)v44, v7, 2);
    if ( v44[0] )
    {
      v9 = sub_15C70A0((__int64)v44);
      v10 = *(_QWORD *)(v9 - 8LL * *(unsigned int *)(v9 + 8));
      v11 = v10;
      if ( *(_BYTE *)v10 == 15 || (v10 = *(_QWORD *)(v10 - 8LL * *(unsigned int *)(v10 + 8)), (v11 = v10) != 0) )
      {
        v12 = *(_QWORD *)(v11 - 8LL * *(unsigned int *)(v10 + 8));
        if ( v12 )
        {
          v13 = (void *)sub_161E970(v12);
          v15 = *(_BYTE **)(v8 + 24);
          v16 = v14;
          if ( *(_QWORD *)(v8 + 16) - (_QWORD)v15 >= v14 )
          {
            if ( v14 )
            {
              memcpy(v15, v13, v14);
              v15 = (_BYTE *)(v16 + *(_QWORD *)(v8 + 24));
              *(_QWORD *)(v8 + 24) = v15;
            }
            goto LABEL_11;
          }
          v8 = sub_16E7EE0(v8, (char *)v13, v14);
        }
      }
      v15 = *(_BYTE **)(v8 + 24);
LABEL_11:
      if ( v15 == *(_BYTE **)(v8 + 16) )
      {
        v8 = sub_16E7EE0(v8, "(", 1u);
      }
      else
      {
        *v15 = 40;
        ++*(_QWORD *)(v8 + 24);
      }
      v17 = sub_15C70B0((__int64)v44);
      v18 = sub_16E7A90(v8, v17);
      v19 = *(_QWORD *)(v18 + 24);
      if ( (unsigned __int64)(*(_QWORD *)(v18 + 16) - v19) <= 2 )
      {
        sub_16E7EE0(v18, "): ", 3u);
      }
      else
      {
        *(_BYTE *)(v19 + 2) = 32;
        *(_WORD *)v19 = 14889;
        *(_QWORD *)(v18 + 24) += 3LL;
      }
      goto LABEL_15;
    }
  }
  v25 = *(void **)(v8 + 24);
  if ( *(_QWORD *)(v8 + 16) - (_QWORD)v25 <= 0xAu )
  {
    v8 = sub_16E7EE0(v8, " Function `", 0xBu);
  }
  else
  {
    qmemcpy(v25, " Function `", 11);
    *(_QWORD *)(v8 + 24) += 11LL;
  }
  v26 = sub_1649960(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 56LL));
  v28 = *(void **)(v8 + 24);
  v29 = (char *)v26;
  v30 = *(_QWORD *)(v8 + 16) - (_QWORD)v28;
  if ( v27 > v30 )
  {
    v41 = sub_16E7EE0(v8, v29, v27);
    v28 = *(void **)(v41 + 24);
    v8 = v41;
    v30 = *(_QWORD *)(v41 + 16) - (_QWORD)v28;
  }
  else if ( v27 )
  {
    v43 = v27;
    memcpy(v28, v29, v27);
    v38 = *(_QWORD *)(v8 + 16);
    v39 = (void *)(*(_QWORD *)(v8 + 24) + v43);
    *(_QWORD *)(v8 + 24) = v39;
    v28 = v39;
    v30 = v38 - (_QWORD)v39;
  }
  if ( v30 <= 0xE )
  {
    v8 = sub_16E7EE0(v8, "' Basic Block `", 0xFu);
  }
  else
  {
    qmemcpy(v28, "' Basic Block `", 15);
    *(_QWORD *)(v8 + 24) += 15LL;
  }
  v31 = sub_1649960(*(_QWORD *)(a2 + 40));
  v33 = *(_BYTE **)(v8 + 24);
  v34 = (char *)v31;
  v35 = *(_QWORD *)(v8 + 16) - (_QWORD)v33;
  if ( v35 < v32 )
  {
    v40 = sub_16E7EE0(v8, v34, v32);
    v33 = *(_BYTE **)(v40 + 24);
    v8 = v40;
    v35 = *(_QWORD *)(v40 + 16) - (_QWORD)v33;
  }
  else if ( v32 )
  {
    v42 = v32;
    memcpy(v33, v34, v32);
    v36 = *(_QWORD *)(v8 + 16);
    v37 = (_BYTE *)(*(_QWORD *)(v8 + 24) + v42);
    *(_QWORD *)(v8 + 24) = v37;
    v33 = v37;
    v35 = v36 - (_QWORD)v37;
  }
  if ( v35 <= 2 )
  {
    sub_16E7EE0(v8, "': ", 3u);
  }
  else
  {
    v33[2] = 32;
    *(_WORD *)v33 = 14887;
    *(_QWORD *)(v8 + 24) += 3LL;
  }
LABEL_15:
  if ( v44[0] )
    sub_161E7C0((__int64)v44, v44[0]);
  v20 = *(_QWORD *)(a1 + 24);
  v21 = *(void **)(v20 + 24);
  if ( *(_QWORD *)(v20 + 16) - (_QWORD)v21 <= 0xBu )
  {
    sub_16E7EE0(v20, "\n  context: ", 0xCu);
  }
  else
  {
    qmemcpy(v21, "\n  context: ", 12);
    *(_QWORD *)(v20 + 24) += 12LL;
  }
  sub_155C2B0(a2, *(_QWORD *)(a1 + 24), 0);
  v22 = *(_QWORD *)(a1 + 24);
  v23 = *(_QWORD *)(v22 + 24);
  if ( (unsigned __int64)(*(_QWORD *)(v22 + 16) - v23) <= 2 )
  {
    sub_16E7EE0(v22, "\n  ", 3u);
  }
  else
  {
    *(_BYTE *)(v23 + 2) = 32;
    *(_WORD *)v23 = 8202;
    *(_QWORD *)(v22 + 24) += 3LL;
  }
  return *(_QWORD *)(a1 + 24);
}
