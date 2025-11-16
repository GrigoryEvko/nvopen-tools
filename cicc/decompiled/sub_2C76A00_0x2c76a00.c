// Function: sub_2C76A00
// Address: 0x2c76a00
//
__int64 __fastcall sub_2C76A00(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v5; // rdi
  _WORD *v6; // rdx
  __int64 v7; // r15
  __int64 v8; // rsi
  __int64 v9; // r12
  __int64 v10; // rax
  unsigned __int8 v11; // dl
  _BYTE **v12; // rax
  _BYTE *v13; // rax
  unsigned __int8 v14; // dl
  unsigned __int8 v15; // dl
  __int64 *v16; // rax
  void *v17; // rax
  size_t v18; // rdx
  _BYTE *v19; // rdi
  __int64 v20; // rax
  unsigned int v21; // eax
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rdi
  void *v25; // rdx
  __int64 v26; // rdi
  __int64 v27; // rdx
  void *v29; // rdx
  const char *v30; // rax
  size_t v31; // rdx
  void *v32; // rdi
  unsigned __int8 *v33; // rsi
  unsigned __int64 v34; // rax
  const char *v35; // rax
  size_t v36; // rdx
  _BYTE *v37; // rdi
  unsigned __int8 *v38; // rsi
  unsigned __int64 v39; // rax
  _BYTE *v40; // rdx
  __int64 v41; // rax
  _BYTE *v42; // rdx
  __int64 v43; // rax
  void *v44; // rdx
  __int64 v45; // rax
  __int64 v46; // rax
  size_t v47; // [rsp+8h] [rbp-48h]
  size_t v48; // [rsp+8h] [rbp-48h]
  size_t v49; // [rsp+8h] [rbp-48h]
  __int64 v50[7]; // [rsp+18h] [rbp-38h] BYREF

  sub_2C763F0(a3, *(_QWORD *)(a1 + 24));
  v5 = *(_QWORD *)(a1 + 24);
  v6 = *(_WORD **)(v5 + 32);
  if ( *(_QWORD *)(v5 + 24) - (_QWORD)v6 <= 1u )
  {
    sub_CB6200(v5, (unsigned __int8 *)": ", 2u);
  }
  else
  {
    *v6 = 8250;
    *(_QWORD *)(v5 + 32) += 2LL;
  }
  v7 = *(_QWORD *)(a1 + 24);
  v8 = *(_QWORD *)(a2 + 48);
  v9 = v7;
  v50[0] = v8;
  if ( v8 )
  {
    sub_B96E90((__int64)v50, v8, 1);
    if ( v50[0] )
    {
      v10 = sub_B10CD0((__int64)v50);
      v11 = *(_BYTE *)(v10 - 16);
      if ( (v11 & 2) != 0 )
        v12 = *(_BYTE ***)(v10 - 32);
      else
        v12 = (_BYTE **)(v10 - 16 - 8LL * ((v11 >> 2) & 0xF));
      v13 = *v12;
      if ( *v13 != 16 )
      {
        v14 = *(v13 - 16);
        if ( (v14 & 2) != 0 )
        {
          v13 = (_BYTE *)**((_QWORD **)v13 - 4);
          if ( !v13 )
            goto LABEL_45;
        }
        else
        {
          v13 = *(_BYTE **)&v13[-8 * ((v14 >> 2) & 0xF) - 16];
          if ( !v13 )
            goto LABEL_45;
        }
      }
      v15 = *(v13 - 16);
      if ( (v15 & 2) != 0 )
        v16 = (__int64 *)*((_QWORD *)v13 - 4);
      else
        v16 = (__int64 *)&v13[-8 * ((v15 >> 2) & 0xF) - 16];
      if ( *v16 )
      {
        v17 = (void *)sub_B91420(*v16);
        v19 = *(_BYTE **)(v7 + 32);
        if ( *(_QWORD *)(v7 + 24) - (_QWORD)v19 >= v18 )
        {
          if ( v18 )
          {
            v47 = v18;
            memcpy(v19, v17, v18);
            v40 = (_BYTE *)(*(_QWORD *)(v7 + 32) + v47);
            *(_QWORD *)(v7 + 32) = v40;
            v19 = v40;
          }
        }
        else
        {
          v20 = sub_CB6200(v7, (unsigned __int8 *)v17, v18);
          v19 = *(_BYTE **)(v20 + 32);
          v9 = v20;
        }
        goto LABEL_15;
      }
LABEL_45:
      v19 = *(_BYTE **)(v7 + 32);
LABEL_15:
      if ( *(_BYTE **)(v9 + 24) == v19 )
      {
        v9 = sub_CB6200(v9, (unsigned __int8 *)"(", 1u);
      }
      else
      {
        *v19 = 40;
        ++*(_QWORD *)(v9 + 32);
      }
      v21 = sub_B10CE0((__int64)v50);
      v22 = sub_CB59D0(v9, v21);
      v23 = *(_QWORD *)(v22 + 32);
      if ( (unsigned __int64)(*(_QWORD *)(v22 + 24) - v23) <= 2 )
      {
        sub_CB6200(v22, (unsigned __int8 *)"): ", 3u);
      }
      else
      {
        *(_BYTE *)(v23 + 2) = 32;
        *(_WORD *)v23 = 14889;
        *(_QWORD *)(v22 + 32) += 3LL;
      }
      goto LABEL_19;
    }
  }
  v29 = *(void **)(v7 + 32);
  if ( *(_QWORD *)(v7 + 24) - (_QWORD)v29 <= 0xAu )
  {
    v9 = sub_CB6200(v7, (unsigned __int8 *)" Function `", 0xBu);
  }
  else
  {
    qmemcpy(v29, " Function `", 11);
    *(_QWORD *)(v7 + 32) += 11LL;
  }
  v30 = sub_BD5D20(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 72LL));
  v32 = *(void **)(v9 + 32);
  v33 = (unsigned __int8 *)v30;
  v34 = *(_QWORD *)(v9 + 24) - (_QWORD)v32;
  if ( v34 < v31 )
  {
    v46 = sub_CB6200(v9, v33, v31);
    v32 = *(void **)(v46 + 32);
    v9 = v46;
    v34 = *(_QWORD *)(v46 + 24) - (_QWORD)v32;
  }
  else if ( v31 )
  {
    v49 = v31;
    memcpy(v32, v33, v31);
    v43 = *(_QWORD *)(v9 + 24);
    v44 = (void *)(*(_QWORD *)(v9 + 32) + v49);
    *(_QWORD *)(v9 + 32) = v44;
    v32 = v44;
    v34 = v43 - (_QWORD)v44;
  }
  if ( v34 <= 0xE )
  {
    v9 = sub_CB6200(v9, "' Basic Block `", 0xFu);
  }
  else
  {
    qmemcpy(v32, "' Basic Block `", 15);
    *(_QWORD *)(v9 + 32) += 15LL;
  }
  v35 = sub_BD5D20(*(_QWORD *)(a2 + 40));
  v37 = *(_BYTE **)(v9 + 32);
  v38 = (unsigned __int8 *)v35;
  v39 = *(_QWORD *)(v9 + 24) - (_QWORD)v37;
  if ( v39 < v36 )
  {
    v45 = sub_CB6200(v9, v38, v36);
    v37 = *(_BYTE **)(v45 + 32);
    v9 = v45;
    v39 = *(_QWORD *)(v45 + 24) - (_QWORD)v37;
  }
  else if ( v36 )
  {
    v48 = v36;
    memcpy(v37, v38, v36);
    v41 = *(_QWORD *)(v9 + 24);
    v42 = (_BYTE *)(*(_QWORD *)(v9 + 32) + v48);
    *(_QWORD *)(v9 + 32) = v42;
    v37 = v42;
    v39 = v41 - (_QWORD)v42;
  }
  if ( v39 <= 2 )
  {
    sub_CB6200(v9, "': ", 3u);
  }
  else
  {
    v37[2] = 32;
    *(_WORD *)v37 = 14887;
    *(_QWORD *)(v9 + 32) += 3LL;
  }
LABEL_19:
  if ( v50[0] )
    sub_B91220((__int64)v50, v50[0]);
  v24 = *(_QWORD *)(a1 + 24);
  v25 = *(void **)(v24 + 32);
  if ( *(_QWORD *)(v24 + 24) - (_QWORD)v25 <= 0xBu )
  {
    sub_CB6200(v24, "\n  context: ", 0xCu);
  }
  else
  {
    qmemcpy(v25, "\n  context: ", 12);
    *(_QWORD *)(v24 + 32) += 12LL;
  }
  sub_A69870(a2, *(_BYTE **)(a1 + 24), 0);
  v26 = *(_QWORD *)(a1 + 24);
  v27 = *(_QWORD *)(v26 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(v26 + 24) - v27) <= 2 )
  {
    sub_CB6200(v26, "\n  ", 3u);
  }
  else
  {
    *(_BYTE *)(v27 + 2) = 32;
    *(_WORD *)v27 = 8202;
    *(_QWORD *)(v26 + 32) += 3LL;
  }
  return *(_QWORD *)(a1 + 24);
}
