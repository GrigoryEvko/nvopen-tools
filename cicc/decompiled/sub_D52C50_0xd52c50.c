// Function: sub_D52C50
// Address: 0xd52c50
//
__int64 __fastcall sub_D52C50(__int64 a1, int *a2)
{
  void *v4; // rdx
  __int64 v5; // r8
  _QWORD **v6; // rdi
  int v7; // r9d
  _QWORD *v8; // rdx
  int v9; // eax
  int v10; // ecx
  int v11; // esi
  _QWORD *v12; // rdx
  int v13; // ecx
  unsigned __int64 v14; // rdx
  _QWORD *v15; // rdx
  __int64 v16; // rax
  __int64 v17; // r8
  _QWORD **v18; // rdi
  _QWORD *v19; // rax
  unsigned __int64 v20; // rsi
  int v21; // edx
  int v22; // ecx
  _QWORD *v23; // rax
  int v24; // edx
  __m128i *v25; // rdx
  __m128i si128; // xmm0
  __int64 v27; // r14
  __int64 v28; // rdi
  void *v29; // rdi
  unsigned __int8 *v30; // rsi
  size_t v31; // r15
  void *v32; // rdx
  __int64 v33; // rbx
  __int64 v34; // r13
  __int64 v35; // rdi
  _BYTE *v36; // rdi
  char *v37; // rsi
  size_t v38; // r15
  __int64 v39; // r8
  _BYTE *v40; // rax
  _BYTE *v41; // rax
  const char *v43; // rax
  size_t v44; // rdx
  const char *v45; // rax
  size_t v46; // rdx

  v4 = *(void **)(a1 + 32);
  if ( *(_QWORD *)(a1 + 24) - (_QWORD)v4 <= 9u )
  {
    sub_CB6200(a1, "IsPerfect=", 0xAu);
    v5 = *(_QWORD *)(a1 + 32);
  }
  else
  {
    qmemcpy(v4, "IsPerfect=", 10);
    v5 = *(_QWORD *)(a1 + 32) + 10LL;
    *(_QWORD *)(a1 + 32) = v5;
  }
  v6 = (_QWORD **)*((_QWORD *)a2 + 1);
  v7 = *a2;
  v8 = (_QWORD *)*v6[(unsigned int)a2[4] - 1];
  if ( v8 )
  {
    v9 = 1;
    do
    {
      v8 = (_QWORD *)*v8;
      v10 = v9++;
    }
    while ( v8 );
    v11 = v10 + 2;
  }
  else
  {
    v11 = 2;
    v9 = 1;
  }
  v12 = (_QWORD *)**v6;
  if ( v12 )
  {
    v13 = 1;
    do
    {
      v12 = (_QWORD *)*v12;
      ++v13;
    }
    while ( v12 );
    v9 = v11 - v13;
  }
  v14 = *(_QWORD *)(a1 + 24) - v5;
  if ( v9 == v7 )
  {
    if ( v14 <= 3 )
    {
      sub_CB6200(a1, (unsigned __int8 *)"true", 4u);
      v15 = *(_QWORD **)(a1 + 32);
    }
    else
    {
      *(_DWORD *)v5 = 1702195828;
      v15 = (_QWORD *)(*(_QWORD *)(a1 + 32) + 4LL);
      *(_QWORD *)(a1 + 32) = v15;
    }
LABEL_16:
    if ( *(_QWORD *)(a1 + 24) - (_QWORD)v15 <= 7u )
      goto LABEL_14;
    goto LABEL_17;
  }
  if ( v14 <= 4 )
  {
    sub_CB6200(a1, (unsigned __int8 *)"false", 5u);
    v15 = *(_QWORD **)(a1 + 32);
    goto LABEL_16;
  }
  *(_DWORD *)v5 = 1936482662;
  *(_BYTE *)(v5 + 4) = 101;
  v15 = (_QWORD *)(*(_QWORD *)(a1 + 32) + 5LL);
  v16 = *(_QWORD *)(a1 + 24);
  *(_QWORD *)(a1 + 32) = v15;
  if ( (unsigned __int64)(v16 - (_QWORD)v15) <= 7 )
  {
LABEL_14:
    v17 = sub_CB6200(a1, ", Depth=", 8u);
    goto LABEL_18;
  }
LABEL_17:
  v17 = a1;
  *v15 = 0x3D6874706544202CLL;
  *(_QWORD *)(a1 + 32) += 8LL;
LABEL_18:
  v18 = (_QWORD **)*((_QWORD *)a2 + 1);
  v19 = (_QWORD *)*v18[(unsigned int)a2[4] - 1];
  if ( v19 )
  {
    LODWORD(v20) = 1;
    do
    {
      v19 = (_QWORD *)*v19;
      v21 = v20;
      v20 = (unsigned int)(v20 + 1);
    }
    while ( v19 );
    v22 = v21 + 2;
  }
  else
  {
    v22 = 2;
    v20 = 1;
  }
  v23 = (_QWORD *)**v18;
  if ( v23 )
  {
    v24 = 1;
    do
    {
      v23 = (_QWORD *)*v23;
      ++v24;
    }
    while ( v23 );
    v20 = (unsigned int)(v22 - v24);
  }
  sub_CB59D0(v17, v20);
  v25 = *(__m128i **)(a1 + 32);
  if ( *(_QWORD *)(a1 + 24) - (_QWORD)v25 <= 0x10u )
  {
    v27 = sub_CB6200(a1, ", OutermostLoop: ", 0x11u);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F71A40);
    v25[1].m128i_i8[0] = 32;
    v27 = a1;
    *v25 = si128;
    *(_QWORD *)(a1 + 32) += 17LL;
  }
  v28 = **(_QWORD **)(**((_QWORD **)a2 + 1) + 32LL);
  if ( !v28 || (*(_BYTE *)(v28 + 7) & 0x10) == 0 )
  {
    v29 = *(void **)(v27 + 32);
    v30 = "<unnamed loop>";
    v31 = 14;
    if ( *(_QWORD *)(v27 + 24) - (_QWORD)v29 <= 0xDu )
    {
LABEL_31:
      sub_CB6200(v27, v30, v31);
      goto LABEL_32;
    }
LABEL_58:
    memcpy(v29, v30, v31);
    *(_QWORD *)(v27 + 32) += v31;
    goto LABEL_32;
  }
  v45 = sub_BD5D20(v28);
  v29 = *(void **)(v27 + 32);
  v30 = (unsigned __int8 *)v45;
  v31 = v46;
  if ( *(_QWORD *)(v27 + 24) - (_QWORD)v29 < v46 )
    goto LABEL_31;
  if ( v46 )
    goto LABEL_58;
LABEL_32:
  v32 = *(void **)(a1 + 32);
  if ( *(_QWORD *)(a1 + 24) - (_QWORD)v32 <= 0xAu )
  {
    sub_CB6200(a1, ", Loops: ( ", 0xBu);
  }
  else
  {
    qmemcpy(v32, ", Loops: ( ", 11);
    *(_QWORD *)(a1 + 32) += 11LL;
  }
  v33 = *((_QWORD *)a2 + 1);
  v34 = v33 + 8LL * (unsigned int)a2[4];
  if ( v34 != v33 )
  {
    while ( 1 )
    {
      v35 = **(_QWORD **)(*(_QWORD *)v33 + 32LL);
      if ( !v35 || (*(_BYTE *)(v35 + 7) & 0x10) == 0 )
        break;
      v43 = sub_BD5D20(v35);
      v36 = *(_BYTE **)(a1 + 32);
      v37 = (char *)v43;
      v40 = *(_BYTE **)(a1 + 24);
      v38 = v44;
      if ( v40 - v36 < v44 )
        goto LABEL_40;
      if ( v44 )
      {
LABEL_49:
        memcpy(v36, v37, v38);
        v40 = *(_BYTE **)(a1 + 24);
        v39 = a1;
        v36 = (_BYTE *)(v38 + *(_QWORD *)(a1 + 32));
        *(_QWORD *)(a1 + 32) = v36;
        goto LABEL_41;
      }
      v39 = a1;
LABEL_41:
      if ( v40 == v36 )
      {
        v33 += 8;
        sub_CB6200(v39, (unsigned __int8 *)" ", 1u);
        if ( v34 == v33 )
          goto LABEL_43;
      }
      else
      {
        v33 += 8;
        *v36 = 32;
        ++*(_QWORD *)(v39 + 32);
        if ( v34 == v33 )
          goto LABEL_43;
      }
    }
    v36 = *(_BYTE **)(a1 + 32);
    v37 = "<unnamed loop>";
    v38 = 14;
    if ( *(_QWORD *)(a1 + 24) - (_QWORD)v36 > 0xDu )
      goto LABEL_49;
LABEL_40:
    v39 = sub_CB6200(a1, (unsigned __int8 *)v37, v38);
    v40 = *(_BYTE **)(v39 + 24);
    v36 = *(_BYTE **)(v39 + 32);
    goto LABEL_41;
  }
LABEL_43:
  v41 = *(_BYTE **)(a1 + 32);
  if ( *(_BYTE **)(a1 + 24) == v41 )
  {
    sub_CB6200(a1, (unsigned __int8 *)")", 1u);
  }
  else
  {
    *v41 = 41;
    ++*(_QWORD *)(a1 + 32);
  }
  return a1;
}
