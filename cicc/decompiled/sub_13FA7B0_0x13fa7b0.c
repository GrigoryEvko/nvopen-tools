// Function: sub_13FA7B0
// Address: 0x13fa7b0
//
_BYTE *__fastcall sub_13FA7B0(_QWORD **a1, __int64 a2, int a3, char a4)
{
  __int64 v4; // r13
  __int64 v6; // rdi
  void *v7; // rax
  _QWORD *v8; // rax
  unsigned int v9; // edx
  __int64 v10; // rsi
  __int64 v11; // rdi
  void *v12; // rax
  __int64 *v13; // rax
  __int64 v14; // r12
  __int64 v15; // r14
  __int64 v16; // rax
  __int64 v17; // rdi
  __int64 v18; // r14
  unsigned int v19; // r12d
  _QWORD **v20; // r13
  __int64 v21; // rax
  _QWORD *v22; // rdx
  __int64 v23; // rbx
  _QWORD *v24; // rax
  _QWORD *v25; // r15
  _QWORD *v26; // rax
  __int64 v27; // rdx
  _QWORD *v28; // rdx
  _BYTE *v29; // rax
  _QWORD *v30; // rdx
  _QWORD *v31; // rdx
  __int64 v32; // rsi
  __int64 v33; // rdx
  _QWORD *v34; // rdx
  _BYTE *v35; // rax
  _BYTE *result; // rax
  __int64 *v37; // r12
  __int64 *i; // rbx
  __int64 v39; // rdi
  __int64 v40; // [rsp+0h] [rbp-60h]
  __int64 v43; // [rsp+10h] [rbp-50h]
  __int64 v44; // [rsp+18h] [rbp-48h]
  _QWORD **v45; // [rsp+20h] [rbp-40h]
  unsigned int v46; // [rsp+28h] [rbp-38h]
  int v47; // [rsp+2Ch] [rbp-34h]

  v4 = a2;
  v6 = sub_16E8750(a2, (unsigned int)(2 * a3));
  v7 = *(void **)(v6 + 24);
  if ( *(_QWORD *)(v6 + 16) - (_QWORD)v7 <= 0xDu )
  {
    v6 = sub_16E7EE0(v6, "Loop at depth ", 14);
    v8 = *a1;
    if ( *a1 )
    {
LABEL_3:
      v9 = 1;
      do
      {
        v8 = (_QWORD *)*v8;
        ++v9;
      }
      while ( v8 );
      v10 = v9;
      goto LABEL_6;
    }
  }
  else
  {
    qmemcpy(v7, "Loop at depth ", 14);
    *(_QWORD *)(v6 + 24) += 14LL;
    v8 = *a1;
    if ( *a1 )
      goto LABEL_3;
  }
  v10 = 1;
LABEL_6:
  v11 = sub_16E7A90(v6, v10);
  v12 = *(void **)(v11 + 24);
  if ( *(_QWORD *)(v11 + 16) - (_QWORD)v12 <= 0xCu )
  {
    sub_16E7EE0(v11, " containing: ", 13);
  }
  else
  {
    qmemcpy(v12, " containing: ", 13);
    *(_QWORD *)(v11 + 24) += 13LL;
  }
  v13 = a1[4];
  v40 = *v13;
  if ( v13 == a1[5] )
    goto LABEL_57;
  v46 = 0;
  v14 = *v13;
  v45 = a1 + 7;
  if ( a4 )
  {
LABEL_30:
    v29 = *(_BYTE **)(v4 + 24);
    if ( *(_BYTE **)(v4 + 16) != v29 )
    {
      *v29 = 10;
      ++*(_QWORD *)(v4 + 24);
      if ( v40 != v14 )
        goto LABEL_13;
LABEL_32:
      v30 = *(_QWORD **)(v4 + 24);
      if ( *(_QWORD *)(v4 + 16) - (_QWORD)v30 <= 7u )
      {
        sub_16E7EE0(v4, "<header>", 8);
      }
      else
      {
        *v30 = 0x3E7265646165683CLL;
        *(_QWORD *)(v4 + 24) += 8LL;
      }
      goto LABEL_13;
    }
    sub_16E7EE0(v4, "\n", 1);
  }
  else
  {
LABEL_10:
    if ( v46 )
    {
      v35 = *(_BYTE **)(v4 + 24);
      if ( *(_BYTE **)(v4 + 16) == v35 )
      {
        sub_16E7EE0(v4, ",", 1);
      }
      else
      {
        *v35 = 44;
        ++*(_QWORD *)(v4 + 24);
      }
    }
    sub_15537D0(v14, v4, 0);
  }
  if ( v40 == v14 )
    goto LABEL_32;
LABEL_13:
  v15 = *(_QWORD *)(*a1[4] + 8LL);
  if ( v15 )
  {
    while ( 1 )
    {
      v16 = sub_1648700(v15);
      if ( (unsigned __int8)(*(_BYTE *)(v16 + 16) - 25) <= 9u )
        break;
      v15 = *(_QWORD *)(v15 + 8);
      if ( !v15 )
        goto LABEL_16;
    }
LABEL_35:
    if ( *(_QWORD *)(v16 + 40) == v14 )
    {
      v33 = *(_QWORD *)(v4 + 24);
      if ( (unsigned __int64)(*(_QWORD *)(v4 + 16) - v33) <= 6 )
      {
        sub_16E7EE0(v4, "<latch>", 7);
      }
      else
      {
        *(_DWORD *)v33 = 1952541756;
        *(_WORD *)(v33 + 4) = 26723;
        *(_BYTE *)(v33 + 6) = 62;
        *(_QWORD *)(v4 + 24) += 7LL;
      }
    }
    else
    {
      while ( 1 )
      {
        v15 = *(_QWORD *)(v15 + 8);
        if ( !v15 )
          break;
        v16 = sub_1648700(v15);
        if ( (unsigned __int8)(*(_BYTE *)(v16 + 16) - 25) <= 9u )
          goto LABEL_35;
      }
    }
  }
LABEL_16:
  v17 = sub_157EBA0(v14);
  if ( !v17 || (v47 = sub_15F4D60(v17), v18 = sub_157EBA0(v14), !v47) )
  {
LABEL_27:
    if ( a4 )
      goto LABEL_52;
    goto LABEL_28;
  }
  v44 = v14;
  v43 = v4;
  v19 = 0;
  v20 = a1;
  do
  {
    v21 = sub_15F4DF0(v18, v19);
    v22 = v20[9];
    v23 = v21;
    v24 = v20[8];
    if ( v22 == v24 )
    {
      v25 = &v24[*((unsigned int *)v20 + 21)];
      if ( v24 == v25 )
      {
        v34 = v20[8];
      }
      else
      {
        do
        {
          if ( v23 == *v24 )
            break;
          ++v24;
        }
        while ( v25 != v24 );
        v34 = v25;
      }
    }
    else
    {
      v25 = &v22[*((unsigned int *)v20 + 20)];
      v24 = (_QWORD *)sub_16CC9F0(v45, v23);
      if ( v23 == *v24 )
      {
        v31 = v20[9];
        if ( v31 == v20[8] )
          v32 = *((unsigned int *)v20 + 21);
        else
          v32 = *((unsigned int *)v20 + 20);
        v34 = &v31[v32];
      }
      else
      {
        v26 = v20[9];
        if ( v26 != v20[8] )
        {
          v24 = &v26[*((unsigned int *)v20 + 20)];
          goto LABEL_24;
        }
        v24 = &v26[*((unsigned int *)v20 + 21)];
        v34 = v24;
      }
    }
    for ( ; v34 != v24; ++v24 )
    {
      if ( *v24 < 0xFFFFFFFFFFFFFFFELL )
        break;
    }
LABEL_24:
    if ( v24 == v25 )
    {
      a1 = v20;
      v4 = v43;
      v14 = v44;
      v27 = *(_QWORD *)(v43 + 24);
      if ( (unsigned __int64)(*(_QWORD *)(v43 + 16) - v27) <= 8 )
      {
        sub_16E7EE0(v43, "<exiting>", 9);
      }
      else
      {
        *(_BYTE *)(v27 + 8) = 62;
        *(_QWORD *)v27 = 0x676E69746978653CLL;
        *(_QWORD *)(v43 + 24) += 9LL;
      }
      goto LABEL_27;
    }
    ++v19;
  }
  while ( v19 != v47 );
  a1 = v20;
  v14 = v44;
  v4 = v43;
  if ( !a4 )
    goto LABEL_28;
LABEL_52:
  sub_155C2B0(v14, v4, 0);
LABEL_28:
  v28 = a1[4];
  if ( ++v46 < (unsigned __int64)(a1[5] - v28) )
  {
    v14 = v28[v46];
    if ( !a4 )
      goto LABEL_10;
    goto LABEL_30;
  }
LABEL_57:
  result = *(_BYTE **)(v4 + 24);
  if ( *(_BYTE **)(v4 + 16) == result )
  {
    result = (_BYTE *)sub_16E7EE0(v4, "\n", 1);
  }
  else
  {
    *result = 10;
    ++*(_QWORD *)(v4 + 24);
  }
  v37 = a1[2];
  for ( i = a1[1]; v37 != i; result = (_BYTE *)sub_13FA7B0(v39, v4, (unsigned int)(a3 + 2), 0) )
    v39 = *i++;
  return result;
}
