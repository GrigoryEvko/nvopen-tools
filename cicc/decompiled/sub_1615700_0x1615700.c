// Function: sub_1615700
// Address: 0x1615700
//
void __fastcall sub_1615700(__int64 a1, const char *a2)
{
  __int64 v3; // rax
  __m128i *v4; // rdx
  void *v5; // rdi
  __int64 v6; // r15
  __int64 v7; // r13
  __int64 v8; // rax
  __int64 v9; // rbx
  __int64 v10; // rax
  _QWORD *v11; // r8
  size_t v12; // rbx
  void **v13; // rbx
  void **v14; // r12
  __int64 v15; // rdi
  _BYTE *v16; // rax
  __int64 v17; // rax
  _QWORD *v18; // [rsp-40h] [rbp-40h]

  if ( dword_4F9EB40 <= 0 )
    return;
  v3 = sub_16BA580(a1, a2, (unsigned int)dword_4F9EB40);
  v4 = *(__m128i **)(v3 + 24);
  v5 = (void *)v3;
  if ( *(_QWORD *)(v3 + 16) - (_QWORD)v4 <= 0xFu )
  {
    a2 = "Pass Arguments: ";
    sub_16E7EE0(v3, "Pass Arguments: ", 16);
  }
  else
  {
    *v4 = _mm_load_si128((const __m128i *)&xmmword_3F552E0);
    *(_QWORD *)(v3 + 24) += 16LL;
  }
  v6 = *(_QWORD *)(a1 + 256);
  v7 = v6 + 8LL * *(unsigned int *)(a1 + 264);
  if ( v6 != v7 )
  {
    while ( 1 )
    {
      v5 = (void *)a1;
      a2 = *(const char **)(*(_QWORD *)v6 + 16LL);
      v8 = sub_1614F20(a1, (__int64)a2);
      v9 = v8;
      if ( !v8 || *(_BYTE *)(v8 + 42) )
        goto LABEL_6;
      v10 = sub_16BA580(a1, a2, v4);
      v4 = *(__m128i **)(v10 + 24);
      v11 = (_QWORD *)v10;
      if ( *(_QWORD *)(v10 + 16) - (_QWORD)v4 <= 1u )
      {
        v17 = sub_16E7EE0(v10, " -", 2);
        v5 = *(void **)(v17 + 24);
        v11 = (_QWORD *)v17;
      }
      else
      {
        v4->m128i_i16[0] = 11552;
        v5 = (void *)(*(_QWORD *)(v10 + 24) + 2LL);
        *(_QWORD *)(v10 + 24) = v5;
      }
      a2 = *(const char **)(v9 + 16);
      v12 = *(_QWORD *)(v9 + 24);
      if ( v12 > v11[2] - (_QWORD)v5 )
        break;
      if ( v12 )
      {
        v18 = v11;
        v6 += 8;
        memcpy(v5, a2, v12);
        v18[3] += v12;
        if ( v7 == v6 )
          goto LABEL_14;
      }
      else
      {
LABEL_6:
        v6 += 8;
        if ( v7 == v6 )
          goto LABEL_14;
      }
    }
    v5 = v11;
    sub_16E7EE0(v11, a2, v12);
    goto LABEL_6;
  }
LABEL_14:
  v13 = *(void ***)(a1 + 32);
  v14 = &v13[*(unsigned int *)(a1 + 40)];
  while ( v14 != v13 )
  {
    v5 = *v13++;
    sub_16155F0((__int64)v5);
  }
  v15 = sub_16BA580(v5, a2, v4);
  v16 = *(_BYTE **)(v15 + 24);
  if ( *(_BYTE **)(v15 + 16) == v16 )
  {
    sub_16E7EE0(v15, "\n", 1);
  }
  else
  {
    *v16 = 10;
    ++*(_QWORD *)(v15 + 24);
  }
}
