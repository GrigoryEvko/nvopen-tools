// Function: sub_B86160
// Address: 0xb86160
//
void __fastcall sub_B86160(__int64 a1, const void *a2, size_t a3, __int64 a4, __int64 a5)
{
  __int64 v9; // rax
  __int64 v10; // r15
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // rax
  void *v14; // rdi
  __int64 v15; // r15
  unsigned __int64 v16; // rax
  void *v17; // rdi
  int v18; // ebx
  __int64 v19; // r12
  __int64 v20; // rdi
  __int64 v21; // r15
  __int64 v22; // r12
  _BYTE *v23; // rax
  const void *v24; // rsi
  size_t v25; // r15
  __int64 v26; // rdi
  _BYTE *v27; // rax
  __int64 v28; // rdi
  _BYTE *v29; // rax
  __int64 v30; // rax
  __m128i *v31; // rdx
  __m128i si128; // xmm0
  __int64 v33; // rax
  __int64 v34; // rax
  _QWORD v35[2]; // [rsp+10h] [rbp-50h] BYREF
  _QWORD v36[8]; // [rsp+20h] [rbp-40h] BYREF

  if ( !*(_DWORD *)(a5 + 8) )
    return;
  v9 = sub_C5F790(a1);
  v35[0] = v36;
  v10 = sub_CB5A80(v9, a4);
  sub_2240A50(v35, (unsigned int)(2 * *(_DWORD *)(a1 + 384) + 3), 32, v11, v12);
  v13 = sub_CB6200(v10, v35[0], v35[1]);
  v14 = *(void **)(v13 + 32);
  v15 = v13;
  v16 = *(_QWORD *)(v13 + 24) - (_QWORD)v14;
  if ( a3 <= v16 )
  {
    if ( a3 )
    {
      memcpy(v14, a2, a3);
      v34 = *(_QWORD *)(v15 + 24);
      v14 = (void *)(a3 + *(_QWORD *)(v15 + 32));
      *(_QWORD *)(v15 + 32) = v14;
      v16 = v34 - (_QWORD)v14;
    }
    if ( v16 > 9 )
      goto LABEL_6;
LABEL_29:
    sub_CB6200(v15, " Analyses:", 10);
    goto LABEL_7;
  }
  v33 = sub_CB6200(v15, a2, a3);
  v14 = *(void **)(v33 + 32);
  v15 = v33;
  if ( *(_QWORD *)(v33 + 24) - (_QWORD)v14 <= 9u )
    goto LABEL_29;
LABEL_6:
  qmemcpy(v14, " Analyses:", 10);
  *(_QWORD *)(v15 + 32) += 10LL;
LABEL_7:
  v17 = (void *)v35[0];
  if ( (_QWORD *)v35[0] != v36 )
    j_j___libc_free_0(v35[0], v36[0] + 1LL);
  v18 = 0;
  v19 = 0;
  if ( *(_DWORD *)(a5 + 8) )
  {
    while ( 1 )
    {
      v20 = *(_QWORD *)(a1 + 8);
      v21 = sub_B85AD0(v20, *(_QWORD *)(*(_QWORD *)a5 + 8 * v19));
      if ( !v21 )
        break;
      v22 = sub_C5F790(v20);
      v23 = *(_BYTE **)(v22 + 32);
      if ( (unsigned __int64)v23 >= *(_QWORD *)(v22 + 24) )
      {
        v22 = sub_CB5D20(v22, 32);
      }
      else
      {
        *(_QWORD *)(v22 + 32) = v23 + 1;
        *v23 = 32;
      }
      v17 = *(void **)(v22 + 32);
      v24 = *(const void **)v21;
      v25 = *(_QWORD *)(v21 + 8);
      if ( v25 <= *(_QWORD *)(v22 + 24) - (_QWORD)v17 )
      {
        if ( v25 )
        {
          memcpy(v17, v24, v25);
          *(_QWORD *)(v22 + 32) += v25;
        }
LABEL_16:
        v19 = (unsigned int)(v18 + 1);
        v18 = v19;
        if ( (_DWORD)v19 == *(_DWORD *)(a5 + 8) )
          goto LABEL_21;
        goto LABEL_17;
      }
      v17 = (void *)v22;
      sub_CB6200(v22, v24, v25);
      v19 = (unsigned int)++v18;
      if ( v18 == *(_DWORD *)(a5 + 8) )
        goto LABEL_21;
LABEL_17:
      if ( v18 )
      {
        v26 = sub_C5F790(v17);
        v27 = *(_BYTE **)(v26 + 32);
        if ( (unsigned __int64)v27 >= *(_QWORD *)(v26 + 24) )
        {
          sub_CB5D20(v26, 44);
        }
        else
        {
          *(_QWORD *)(v26 + 32) = v27 + 1;
          *v27 = 44;
        }
      }
    }
    v30 = sub_C5F790(v20);
    v31 = *(__m128i **)(v30 + 32);
    v17 = (void *)v30;
    if ( *(_QWORD *)(v30 + 24) - (_QWORD)v31 <= 0x12u )
    {
      sub_CB6200(v30, " Uninitialized Pass", 19);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F552F0);
      v31[1].m128i_i8[2] = 115;
      v31[1].m128i_i16[0] = 29537;
      *v31 = si128;
      *(_QWORD *)(v30 + 32) += 19LL;
    }
    goto LABEL_16;
  }
LABEL_21:
  v28 = sub_C5F790(v17);
  v29 = *(_BYTE **)(v28 + 32);
  if ( (unsigned __int64)v29 >= *(_QWORD *)(v28 + 24) )
  {
    sub_CB5D20(v28, 10);
  }
  else
  {
    *(_QWORD *)(v28 + 32) = v29 + 1;
    *v29 = 10;
  }
}
