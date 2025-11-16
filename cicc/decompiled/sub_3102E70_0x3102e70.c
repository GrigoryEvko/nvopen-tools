// Function: sub_3102E70
// Address: 0x3102e70
//
void __fastcall sub_3102E70(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v6; // eax
  __int64 v7; // r8
  int v8; // ecx
  int v9; // r11d
  unsigned int v11; // eax
  unsigned int v12; // r10d
  __int64 v13; // r13
  __int64 v14; // rdx
  __int64 v15; // rdi
  unsigned int v16; // ebx
  __m128i *v17; // rdx
  __int64 v18; // rax
  __m128i si128; // xmm0
  char *v20; // rbx
  char v21; // r13
  __int64 v22; // r14
  unsigned __int8 *v23; // rax
  size_t v24; // rdx
  void *v25; // rdi
  __int64 v26; // r12
  _WORD *v27; // rdx
  __int64 v28; // rdi
  __int64 v29; // rax
  unsigned int v30; // edx
  unsigned int v31; // eax
  __int64 v32; // rcx
  __m128i *v33; // rsi
  unsigned __int64 v34; // rcx
  __int64 v35; // rdi
  __int64 v36; // rax
  _QWORD *v37; // rdx
  _BYTE *v38; // rax
  size_t v39; // rdx
  __int64 v40; // rcx
  int v41; // esi
  size_t v42; // [rsp-80h] [rbp-80h]
  char *v43; // [rsp-70h] [rbp-70h]
  char *v44; // [rsp-68h] [rbp-68h] BYREF
  __int64 v45; // [rsp-60h] [rbp-60h]
  _BYTE v46[88]; // [rsp-58h] [rbp-58h] BYREF

  v6 = *(_DWORD *)(a1 + 32);
  v7 = *(_QWORD *)(a1 + 16);
  if ( !v6 )
    return;
  v8 = v6 - 1;
  v9 = 1;
  v11 = (v6 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v12 = v8 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v13 = v7 + 56LL * v11;
  v14 = *(_QWORD *)v13;
  v15 = *(_QWORD *)v13;
  if ( a2 == *(_QWORD *)v13 )
  {
LABEL_3:
    v16 = *(_DWORD *)(v13 + 16);
    v44 = v46;
    v45 = 0x400000000LL;
    if ( !v16 || &v44 == (char **)(v13 + 8) )
    {
      v17 = *(__m128i **)(a3 + 32);
      v18 = *(_QWORD *)(a3 + 24);
    }
    else
    {
      if ( v16 > 4 )
      {
        sub_C8D5F0((__int64)&v44, v46, v16, 8u, v7, a6);
        v39 = 8LL * *(unsigned int *)(v13 + 16);
        if ( v39 )
          memcpy(v44, *(const void **)(v13 + 8), v39);
        v33 = *(__m128i **)(a3 + 32);
        v40 = *(_QWORD *)(a3 + 24);
        LODWORD(v45) = v16;
        v34 = v40 - (_QWORD)v33;
        goto LABEL_23;
      }
      v28 = *(_QWORD *)(v13 + 8);
      v29 = 8 * v16;
      *(_QWORD *)&v46[v29 - 8] = *(_QWORD *)(v28 + v29 - 8);
      if ( (unsigned int)(v29 - 1) >= 8 )
      {
        v30 = (v29 - 1) & 0xFFFFFFF8;
        v31 = 0;
        do
        {
          v32 = v31;
          v31 += 8;
          *(_QWORD *)&v46[v32] = *(_QWORD *)(v28 + v32);
        }
        while ( v31 < v30 );
      }
      v18 = *(_QWORD *)(a3 + 24);
      v17 = *(__m128i **)(a3 + 32);
      LODWORD(v45) = v16;
      v33 = v17;
      v34 = v18 - (_QWORD)v17;
      if ( v16 != 1 )
      {
LABEL_23:
        if ( v34 <= 0xF )
        {
          v35 = sub_CB6200(a3, " ; (mustexec in ", 0x10u);
        }
        else
        {
          v35 = a3;
          *v33 = _mm_load_si128((const __m128i *)&xmmword_428C370);
          *(_QWORD *)(a3 + 32) += 16LL;
        }
        v36 = sub_CB59D0(v35, v16);
        v37 = *(_QWORD **)(v36 + 32);
        if ( *(_QWORD *)(v36 + 24) - (_QWORD)v37 <= 7u )
        {
          sub_CB6200(v36, " loops: ", 8u);
        }
        else
        {
          *v37 = 0x203A73706F6F6C20LL;
          *(_QWORD *)(v36 + 32) += 8LL;
        }
        goto LABEL_7;
      }
    }
LABEL_5:
    if ( (unsigned __int64)(v18 - (_QWORD)v17) <= 0x10 )
    {
      sub_CB6200(a3, " ; (mustexec in: ", 0x11u);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_428C380);
      v17[1].m128i_i8[0] = 32;
      *v17 = si128;
      *(_QWORD *)(a3 + 32) += 17LL;
    }
LABEL_7:
    v20 = v44;
    v21 = 1;
    v43 = &v44[8 * (unsigned int)v45];
    if ( v43 != v44 )
    {
      do
      {
        while ( 1 )
        {
          v26 = *(_QWORD *)v20;
          if ( v21 )
          {
            v22 = a3;
            v21 = 0;
          }
          else
          {
            v27 = *(_WORD **)(a3 + 32);
            if ( *(_QWORD *)(a3 + 24) - (_QWORD)v27 > 1u )
            {
              v22 = a3;
              *v27 = 8236;
              *(_QWORD *)(a3 + 32) += 2LL;
            }
            else
            {
              v22 = sub_CB6200(a3, (unsigned __int8 *)", ", 2u);
            }
          }
          v23 = (unsigned __int8 *)sub_BD5D20(**(_QWORD **)(v26 + 32));
          v25 = *(void **)(v22 + 32);
          if ( v24 <= *(_QWORD *)(v22 + 24) - (_QWORD)v25 )
            break;
          v20 += 8;
          sub_CB6200(v22, v23, v24);
          if ( v43 == v20 )
            goto LABEL_32;
        }
        if ( v24 )
        {
          v42 = v24;
          memcpy(v25, v23, v24);
          *(_QWORD *)(v22 + 32) += v42;
        }
        v20 += 8;
      }
      while ( v43 != v20 );
    }
LABEL_32:
    v38 = *(_BYTE **)(a3 + 32);
    if ( *(_BYTE **)(a3 + 24) == v38 )
    {
      sub_CB6200(a3, (unsigned __int8 *)")", 1u);
    }
    else
    {
      *v38 = 41;
      ++*(_QWORD *)(a3 + 32);
    }
    if ( v44 != v46 )
      _libc_free((unsigned __int64)v44);
    return;
  }
  while ( v15 != -4096 )
  {
    v12 = v8 & (v9 + v12);
    v15 = *(_QWORD *)(v7 + 56LL * v12);
    if ( a2 == v15 )
    {
      v41 = 1;
      while ( v14 != -4096 )
      {
        v11 = v8 & (v41 + v11);
        v13 = v7 + 56LL * v11;
        v14 = *(_QWORD *)v13;
        if ( v15 == *(_QWORD *)v13 )
          goto LABEL_3;
        ++v41;
      }
      v17 = *(__m128i **)(a3 + 32);
      v44 = v46;
      v45 = 0x400000000LL;
      v18 = *(_QWORD *)(a3 + 24);
      goto LABEL_5;
    }
    ++v9;
  }
}
