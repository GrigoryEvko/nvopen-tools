// Function: sub_1436AC0
// Address: 0x1436ac0
//
void __fastcall sub_1436AC0(__int64 a1, __int64 a2, __int64 a3)
{
  int v3; // eax
  int v4; // ecx
  int v5; // r11d
  unsigned int v7; // eax
  unsigned int v8; // r10d
  __int64 v9; // r9
  __int64 *v10; // r13
  __int64 v11; // rdx
  __int64 v12; // rdi
  unsigned int v13; // ebx
  __int64 v14; // rax
  __m128i *v15; // rdx
  __m128i si128; // xmm0
  __int64 v17; // rdi
  __int64 v18; // rax
  unsigned int v19; // edx
  unsigned int v20; // eax
  __int64 v21; // rcx
  __m128i *v22; // rsi
  unsigned __int64 v23; // rcx
  __int64 v24; // rdi
  __int64 v25; // rax
  _QWORD *v26; // rdx
  char *v27; // r13
  __int64 v28; // r12
  char *i; // rbx
  __int64 v30; // rax
  size_t v31; // rdx
  char *v32; // rdi
  const char *v33; // rsi
  char *v34; // rax
  size_t v35; // r12
  size_t v36; // rdx
  __int64 v37; // rcx
  int v38; // esi
  char *v39; // [rsp-68h] [rbp-68h] BYREF
  __int64 v40; // [rsp-60h] [rbp-60h]
  _BYTE v41[88]; // [rsp-58h] [rbp-58h] BYREF

  v3 = *(_DWORD *)(a1 + 32);
  if ( !v3 )
    return;
  v4 = v3 - 1;
  v5 = 1;
  v7 = (v3 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v8 = v7;
  v9 = *(_QWORD *)(a1 + 16);
  v10 = (__int64 *)(v9 + 56LL * v7);
  v11 = *v10;
  v12 = *v10;
  if ( a2 == *v10 )
  {
LABEL_3:
    v13 = *((_DWORD *)v10 + 4);
    v39 = v41;
    v40 = 0x400000000LL;
    if ( !v13 || &v39 == (char **)(v10 + 1) )
    {
      v14 = *(_QWORD *)(a3 + 16);
      v15 = *(__m128i **)(a3 + 24);
    }
    else
    {
      if ( v13 > 4 )
      {
        sub_16CD150(&v39, v41, v13, 8);
        v36 = 8LL * *((unsigned int *)v10 + 4);
        if ( v36 )
          memcpy(v39, (const void *)v10[1], v36);
        v22 = *(__m128i **)(a3 + 24);
        v37 = *(_QWORD *)(a3 + 16);
        LODWORD(v40) = v13;
        v23 = v37 - (_QWORD)v22;
        goto LABEL_14;
      }
      v17 = v10[1];
      v18 = 8 * v13;
      *(_QWORD *)&v41[v18 - 8] = *(_QWORD *)(v17 + v18 - 8);
      if ( (unsigned int)(v18 - 1) >= 8 )
      {
        v19 = (v18 - 1) & 0xFFFFFFF8;
        v20 = 0;
        do
        {
          v21 = v20;
          v20 += 8;
          *(_QWORD *)&v41[v21] = *(_QWORD *)(v17 + v21);
        }
        while ( v20 < v19 );
      }
      v14 = *(_QWORD *)(a3 + 16);
      v15 = *(__m128i **)(a3 + 24);
      LODWORD(v40) = v13;
      v22 = v15;
      v23 = v14 - (_QWORD)v15;
      if ( v13 != 1 )
      {
LABEL_14:
        if ( v23 <= 0xF )
        {
          v24 = sub_16E7EE0(a3, " ; (mustexec in ", 16);
        }
        else
        {
          v24 = a3;
          *v22 = _mm_load_si128((const __m128i *)&xmmword_428C370);
          *(_QWORD *)(a3 + 24) += 16LL;
        }
        v25 = sub_16E7A90(v24, v13);
        v26 = *(_QWORD **)(v25 + 24);
        if ( *(_QWORD *)(v25 + 16) - (_QWORD)v26 <= 7u )
        {
          sub_16E7EE0(v25, " loops: ", 8);
        }
        else
        {
          *v26 = 0x203A73706F6F6C20LL;
          *(_QWORD *)(v25 + 24) += 8LL;
        }
        goto LABEL_19;
      }
    }
LABEL_5:
    if ( (unsigned __int64)(v14 - (_QWORD)v15) <= 0x10 )
    {
      sub_16E7EE0(a3, " ; (mustexec in: ", 17);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_428C380);
      v15[1].m128i_i8[0] = 32;
      *v15 = si128;
      *(_QWORD *)(a3 + 24) += 17LL;
    }
LABEL_19:
    v27 = &v39[8 * (unsigned int)v40];
    if ( v27 != v39 )
    {
      v28 = *(_QWORD *)v39;
      for ( i = v39 + 8; ; i += 8 )
      {
        v30 = sub_1649960(**(_QWORD **)(v28 + 32));
        v32 = *(char **)(a3 + 24);
        v33 = (const char *)v30;
        v34 = *(char **)(a3 + 16);
        v35 = v31;
        if ( v31 > v34 - v32 )
        {
          sub_16E7EE0(a3, v33, v31);
          v34 = *(char **)(a3 + 16);
          v32 = *(char **)(a3 + 24);
          if ( v27 == i )
            goto LABEL_32;
        }
        else
        {
          if ( v31 )
          {
            memcpy(v32, v33, v31);
            v34 = *(char **)(a3 + 16);
            v32 = (char *)(v35 + *(_QWORD *)(a3 + 24));
            *(_QWORD *)(a3 + 24) = v32;
          }
          if ( v27 == i )
            goto LABEL_32;
        }
        v28 = *(_QWORD *)i;
        if ( (unsigned __int64)(v34 - v32) <= 1 )
        {
          sub_16E7EE0(a3, ", ", 2);
        }
        else
        {
          *(_WORD *)v32 = 8236;
          *(_QWORD *)(a3 + 24) += 2LL;
        }
      }
    }
    v34 = *(char **)(a3 + 16);
    v32 = *(char **)(a3 + 24);
LABEL_32:
    if ( v32 == v34 )
    {
      sub_16E7EE0(a3, ")", 1);
    }
    else
    {
      *v32 = 41;
      ++*(_QWORD *)(a3 + 24);
    }
    if ( v39 != v41 )
      _libc_free((unsigned __int64)v39);
    return;
  }
  while ( v12 != -8 )
  {
    v8 = v4 & (v5 + v8);
    v12 = *(_QWORD *)(v9 + 56LL * v8);
    if ( a2 == v12 )
    {
      v38 = 1;
      while ( v11 != -8 )
      {
        v7 = v4 & (v38 + v7);
        v10 = (__int64 *)(v9 + 56LL * v7);
        v11 = *v10;
        if ( v12 == *v10 )
          goto LABEL_3;
        ++v38;
      }
      v15 = *(__m128i **)(a3 + 24);
      v40 = 0x400000000LL;
      v14 = *(_QWORD *)(a3 + 16);
      v39 = v41;
      goto LABEL_5;
    }
    ++v5;
  }
}
