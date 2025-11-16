// Function: sub_1615A50
// Address: 0x1615a50
//
void __fastcall sub_1615A50(__int64 a1, const char *a2, size_t a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rsi
  __int64 v10; // rax
  __int64 v11; // r15
  __int64 v12; // rcx
  __int64 v13; // r8
  const char *v14; // rsi
  __int64 v15; // rax
  __int64 v16; // rcx
  __int64 v17; // r8
  const char *v18; // r9
  void *v19; // rdi
  __int64 v20; // r15
  unsigned __int64 v21; // rax
  char *v22; // rdi
  __m128i *v23; // rdx
  int v24; // ebx
  __int64 v25; // r12
  __int64 v26; // rdi
  __int64 v27; // rdx
  __int64 v28; // r15
  __int64 v29; // r12
  _BYTE *v30; // rax
  size_t v31; // r15
  __int64 v32; // rdi
  _BYTE *v33; // rax
  __int64 v34; // rdi
  _BYTE *v35; // rax
  __int64 v36; // rax
  __m128i si128; // xmm0
  __int64 v38; // rax
  __int64 v39; // rax
  const char *v41[2]; // [rsp+10h] [rbp-50h] BYREF
  _QWORD v42[8]; // [rsp+20h] [rbp-40h] BYREF

  v5 = *(unsigned int *)(a5 + 8);
  if ( !(_DWORD)v5 )
    return;
  v10 = sub_16BA580(a1, v5, a3);
  v41[0] = (const char *)v42;
  v11 = sub_16E7B40(v10, a4);
  sub_2240A50(v41, (unsigned int)(2 * *(_DWORD *)(a1 + 400) + 3), 32, v12, v13);
  v14 = v41[0];
  v15 = sub_16E7EE0(v11, v41[0], v41[1]);
  v18 = a2;
  v19 = *(void **)(v15 + 24);
  v20 = v15;
  v21 = *(_QWORD *)(v15 + 16) - (_QWORD)v19;
  if ( a3 <= v21 )
  {
    if ( a3 )
    {
      v14 = a2;
      memcpy(v19, a2, a3);
      v39 = *(_QWORD *)(v20 + 16);
      v19 = (void *)(a3 + *(_QWORD *)(v20 + 24));
      *(_QWORD *)(v20 + 24) = v19;
      v21 = v39 - (_QWORD)v19;
    }
    if ( v21 > 9 )
      goto LABEL_6;
LABEL_29:
    v14 = " Analyses:";
    sub_16E7EE0(v20, " Analyses:", 10, v16, v17, v18);
    goto LABEL_7;
  }
  v14 = a2;
  v38 = sub_16E7EE0(v20, a2, a3);
  v19 = *(void **)(v38 + 24);
  v20 = v38;
  if ( *(_QWORD *)(v38 + 16) - (_QWORD)v19 <= 9u )
    goto LABEL_29;
LABEL_6:
  qmemcpy(v19, " Analyses:", 10);
  *(_QWORD *)(v20 + 24) += 10LL;
LABEL_7:
  v22 = (char *)v41[0];
  if ( (_QWORD *)v41[0] != v42 )
  {
    v14 = (const char *)(v42[0] + 1LL);
    j_j___libc_free_0(v41[0], v42[0] + 1LL);
  }
  v23 = (__m128i *)*(unsigned int *)(a5 + 8);
  v24 = 0;
  v25 = 0;
  if ( (_DWORD)v23 )
  {
    while ( 1 )
    {
      v26 = *(_QWORD *)(a1 + 16);
      v14 = *(const char **)(*(_QWORD *)a5 + 8 * v25);
      v28 = sub_1614F20(v26, (__int64)v14);
      if ( !v28 )
        break;
      v29 = sub_16BA580(v26, v14, v27);
      v30 = *(_BYTE **)(v29 + 24);
      if ( (unsigned __int64)v30 >= *(_QWORD *)(v29 + 16) )
      {
        v29 = sub_16E7DE0(v29, 32);
      }
      else
      {
        *(_QWORD *)(v29 + 24) = v30 + 1;
        *v30 = 32;
      }
      v22 = *(char **)(v29 + 24);
      v14 = *(const char **)v28;
      v31 = *(_QWORD *)(v28 + 8);
      if ( v31 <= *(_QWORD *)(v29 + 16) - (_QWORD)v22 )
      {
        if ( v31 )
        {
          memcpy(v22, v14, v31);
          *(_QWORD *)(v29 + 24) += v31;
        }
LABEL_16:
        v25 = (unsigned int)(v24 + 1);
        v24 = v25;
        if ( (_DWORD)v25 == *(_DWORD *)(a5 + 8) )
          goto LABEL_21;
        goto LABEL_17;
      }
      v22 = (char *)v29;
      sub_16E7EE0(v29, v14, v31);
      v25 = (unsigned int)++v24;
      if ( v24 == *(_DWORD *)(a5 + 8) )
        goto LABEL_21;
LABEL_17:
      if ( v24 )
      {
        v32 = sub_16BA580(v22, v14, v23);
        v33 = *(_BYTE **)(v32 + 24);
        if ( (unsigned __int64)v33 >= *(_QWORD *)(v32 + 16) )
        {
          sub_16E7DE0(v32, 44);
        }
        else
        {
          *(_QWORD *)(v32 + 24) = v33 + 1;
          *v33 = 44;
        }
      }
    }
    v36 = sub_16BA580(v26, v14, v27);
    v23 = *(__m128i **)(v36 + 24);
    v22 = (char *)v36;
    if ( *(_QWORD *)(v36 + 16) - (_QWORD)v23 <= 0x12u )
    {
      v14 = " Uninitialized Pass";
      sub_16E7EE0(v36, " Uninitialized Pass", 19);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F552F0);
      v23[1].m128i_i8[2] = 115;
      v23[1].m128i_i16[0] = 29537;
      *v23 = si128;
      *(_QWORD *)(v36 + 24) += 19LL;
    }
    goto LABEL_16;
  }
LABEL_21:
  v34 = sub_16BA580(v22, v14, v23);
  v35 = *(_BYTE **)(v34 + 24);
  if ( (unsigned __int64)v35 >= *(_QWORD *)(v34 + 16) )
  {
    sub_16E7DE0(v34, 10);
  }
  else
  {
    *(_QWORD *)(v34 + 24) = v35 + 1;
    *v35 = 10;
  }
}
