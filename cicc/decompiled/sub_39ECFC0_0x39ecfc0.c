// Function: sub_39ECFC0
// Address: 0x39ecfc0
//
void __fastcall sub_39ECFC0(__int64 a1, unsigned int a2, unsigned int a3, unsigned int a4, _BYTE *a5, _BYTE *a6)
{
  __int64 v11; // rdi
  __m128i *v12; // rdx
  __m128i si128; // xmm0
  __int64 v14; // rdi
  _BYTE *v15; // rax
  __int64 v16; // rdi
  _BYTE *v17; // rax
  __int64 v18; // rdi
  _BYTE *v19; // rax
  __int64 v20; // rdi
  _BYTE *v21; // rax
  size_t v22; // rdx
  __int64 v23; // rdi
  _BYTE *v24; // rax
  __int64 v25; // r8
  char *v26; // rsi
  void *v27; // rdi
  __int64 v28; // [rsp+8h] [rbp-48h]
  __int64 v29; // [rsp+10h] [rbp-40h]

  v11 = *(_QWORD *)(a1 + 272);
  v12 = *(__m128i **)(v11 + 24);
  if ( *(_QWORD *)(v11 + 16) - (_QWORD)v12 <= 0x15u )
  {
    v11 = sub_16E7EE0(v11, "\t.cv_inline_linetable\t", 0x16u);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F7FA30);
    v12[1].m128i_i32[0] = 1818386804;
    v12[1].m128i_i16[2] = 2405;
    *v12 = si128;
    *(_QWORD *)(v11 + 24) += 22LL;
  }
  v14 = sub_16E7A90(v11, a2);
  v15 = *(_BYTE **)(v14 + 24);
  if ( (unsigned __int64)v15 >= *(_QWORD *)(v14 + 16) )
  {
    v14 = sub_16E7DE0(v14, 32);
  }
  else
  {
    *(_QWORD *)(v14 + 24) = v15 + 1;
    *v15 = 32;
  }
  v16 = sub_16E7A90(v14, a3);
  v17 = *(_BYTE **)(v16 + 24);
  if ( (unsigned __int64)v17 >= *(_QWORD *)(v16 + 16) )
  {
    v16 = sub_16E7DE0(v16, 32);
  }
  else
  {
    *(_QWORD *)(v16 + 24) = v17 + 1;
    *v17 = 32;
  }
  v18 = sub_16E7A90(v16, a4);
  v19 = *(_BYTE **)(v18 + 24);
  if ( (unsigned __int64)v19 >= *(_QWORD *)(v18 + 16) )
  {
    sub_16E7DE0(v18, 32);
  }
  else
  {
    *(_QWORD *)(v18 + 24) = v19 + 1;
    *v19 = 32;
  }
  sub_38E2490(a5, *(_QWORD *)(a1 + 272), *(_BYTE **)(a1 + 280));
  v20 = *(_QWORD *)(a1 + 272);
  v21 = *(_BYTE **)(v20 + 24);
  if ( (unsigned __int64)v21 >= *(_QWORD *)(v20 + 16) )
  {
    sub_16E7DE0(v20, 32);
  }
  else
  {
    *(_QWORD *)(v20 + 24) = v21 + 1;
    *v21 = 32;
  }
  sub_38E2490(a6, *(_QWORD *)(a1 + 272), *(_BYTE **)(a1 + 280));
  v22 = *(unsigned int *)(a1 + 312);
  if ( *(_DWORD *)(a1 + 312) )
  {
    v25 = *(_QWORD *)(a1 + 272);
    v26 = *(char **)(a1 + 304);
    v27 = *(void **)(v25 + 24);
    if ( v22 > *(_QWORD *)(v25 + 16) - (_QWORD)v27 )
    {
      sub_16E7EE0(*(_QWORD *)(a1 + 272), v26, v22);
    }
    else
    {
      v28 = *(_QWORD *)(a1 + 272);
      v29 = *(unsigned int *)(a1 + 312);
      memcpy(v27, v26, v22);
      *(_QWORD *)(v28 + 24) += v29;
    }
  }
  *(_DWORD *)(a1 + 312) = 0;
  if ( (*(_BYTE *)(a1 + 680) & 1) != 0 )
  {
    sub_39E0440(a1);
  }
  else
  {
    v23 = *(_QWORD *)(a1 + 272);
    v24 = *(_BYTE **)(v23 + 24);
    if ( (unsigned __int64)v24 >= *(_QWORD *)(v23 + 16) )
    {
      sub_16E7DE0(v23, 10);
    }
    else
    {
      *(_QWORD *)(v23 + 24) = v24 + 1;
      *v24 = 10;
    }
  }
  nullsub_1942();
}
