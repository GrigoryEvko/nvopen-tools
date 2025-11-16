// Function: sub_30B0F90
// Address: 0x30b0f90
//
__int64 __fastcall sub_30B0F90(__int64 a1, unsigned __int64 a2)
{
  __int64 v3; // r13
  void *v4; // rdx
  __int64 v5; // rdi
  _BYTE *v6; // rax
  __int64 v7; // rdi
  _BYTE *v8; // rax
  int v9; // eax
  void *v10; // rdx
  __int64 *v11; // rcx
  __int64 *v12; // r15
  __int64 v13; // rbx
  __int64 v14; // r12
  _BYTE *v15; // rax
  char *v16; // r12
  size_t v17; // rdx
  _QWORD *v18; // rax
  __int64 v19; // rbx
  __int64 v20; // r12
  __int64 v21; // rax
  unsigned __int64 v23; // rdi
  char *v24; // rax
  char *v25; // r12
  unsigned int v26; // eax
  unsigned int v27; // eax
  unsigned int v28; // ecx
  __int64 v29; // rsi
  __m128i *v30; // rdx
  __m128i si128; // xmm0
  _QWORD *v32; // rbx
  _QWORD *v33; // r12
  int v34; // r15d
  __int64 v35; // rax
  __int64 v36; // rdi
  _BYTE *v37; // rax
  __m128i *v38; // rdx
  __m128i v39; // xmm0
  __int64 *v40; // [rsp+8h] [rbp-38h]

  v3 = a1;
  v4 = *(void **)(a1 + 32);
  if ( *(_QWORD *)(a1 + 24) - (_QWORD)v4 <= 0xCu )
  {
    a1 = sub_CB6200(a1, "Node Address:", 0xDu);
  }
  else
  {
    qmemcpy(v4, "Node Address:", 13);
    *(_QWORD *)(a1 + 32) += 13LL;
  }
  v5 = sub_CB5A80(a1, a2);
  v6 = *(_BYTE **)(v5 + 32);
  if ( *(_BYTE **)(v5 + 24) == v6 )
  {
    v5 = sub_CB6200(v5, (unsigned __int8 *)":", 1u);
  }
  else
  {
    *v6 = 58;
    ++*(_QWORD *)(v5 + 32);
  }
  v7 = sub_30B0C30(v5, *(_DWORD *)(a2 + 56));
  v8 = *(_BYTE **)(v7 + 32);
  if ( *(_BYTE **)(v7 + 24) == v8 )
  {
    sub_CB6200(v7, (unsigned __int8 *)"\n", 1u);
  }
  else
  {
    *v8 = 10;
    ++*(_QWORD *)(v7 + 32);
  }
  v9 = *(_DWORD *)(a2 + 56);
  if ( (unsigned int)(v9 - 1) > 1 )
  {
    if ( v9 == 3 )
    {
      v30 = *(__m128i **)(v3 + 32);
      if ( *(_QWORD *)(v3 + 24) - (_QWORD)v30 <= 0x22u )
      {
        sub_CB6200(v3, "--- start of nodes in pi-block ---\n", 0x23u);
      }
      else
      {
        si128 = _mm_load_si128((const __m128i *)&xmmword_44CBAD0);
        v30[2].m128i_i8[2] = 10;
        v30[2].m128i_i16[0] = 11565;
        *v30 = si128;
        v30[1] = _mm_load_si128((const __m128i *)&xmmword_44CBAE0);
        *(_QWORD *)(v3 + 32) += 35LL;
      }
      v32 = *(_QWORD **)(a2 + 64);
      v33 = &v32[*(unsigned int *)(a2 + 72)];
      if ( v33 != v32 )
      {
        v34 = 0;
        do
        {
          ++v34;
          v35 = sub_30B0F90(v3, *v32);
          v36 = v35;
          if ( v34 != *(_DWORD *)(a2 + 72) )
          {
            v37 = *(_BYTE **)(v35 + 32);
            if ( v37 == *(_BYTE **)(v36 + 24) )
            {
              sub_CB6200(v36, (unsigned __int8 *)"\n", 1u);
            }
            else
            {
              *v37 = 10;
              ++*(_QWORD *)(v36 + 32);
            }
          }
          ++v32;
        }
        while ( v33 != v32 );
      }
      v38 = *(__m128i **)(v3 + 32);
      if ( *(_QWORD *)(v3 + 24) - (_QWORD)v38 <= 0x20u )
      {
        sub_CB6200(v3, "--- end of nodes in pi-block ---\n", 0x21u);
      }
      else
      {
        v39 = _mm_load_si128((const __m128i *)&xmmword_44CBAF0);
        v38[2].m128i_i8[0] = 10;
        *v38 = v39;
        v38[1] = _mm_load_si128((const __m128i *)&xmmword_44CBB00);
        *(_QWORD *)(v3 + 32) += 33LL;
      }
    }
    else if ( v9 != 4 )
    {
      BUG();
    }
  }
  else
  {
    v10 = *(void **)(v3 + 32);
    if ( *(_QWORD *)(v3 + 24) - (_QWORD)v10 <= 0xEu )
    {
      sub_CB6200(v3, " Instructions:\n", 0xFu);
    }
    else
    {
      qmemcpy(v10, " Instructions:\n", 15);
      *(_QWORD *)(v3 + 32) += 15LL;
    }
    v11 = *(__int64 **)(a2 + 64);
    v40 = &v11[*(unsigned int *)(a2 + 72)];
    if ( v11 != v40 )
    {
      v12 = *(__int64 **)(a2 + 64);
      do
      {
        v13 = *v12;
        v14 = sub_CB69B0(v3, 2u);
        sub_A69870(v13, (_BYTE *)v14, 0);
        v15 = *(_BYTE **)(v14 + 32);
        if ( *(_BYTE **)(v14 + 24) == v15 )
        {
          sub_CB6200(v14, (unsigned __int8 *)"\n", 1u);
        }
        else
        {
          *v15 = 10;
          ++*(_QWORD *)(v14 + 32);
        }
        ++v12;
      }
      while ( v40 != v12 );
    }
  }
  v16 = " Edges:none!\n";
  if ( *(_DWORD *)(a2 + 48) )
    v16 = " Edges:\n";
  v17 = strlen(v16);
  v18 = *(_QWORD **)(v3 + 32);
  if ( v17 <= *(_QWORD *)(v3 + 24) - (_QWORD)v18 )
  {
    v23 = (unsigned __int64)(v18 + 1) & 0xFFFFFFFFFFFFFFF8LL;
    *v18 = *(_QWORD *)v16;
    *(_QWORD *)((char *)v18 + (unsigned int)v17 - 8) = *(_QWORD *)&v16[(unsigned int)v17 - 8];
    v24 = (char *)v18 - v23;
    v25 = (char *)(v16 - v24);
    v26 = (v17 + (_DWORD)v24) & 0xFFFFFFF8;
    if ( v26 >= 8 )
    {
      v27 = v26 & 0xFFFFFFF8;
      v28 = 0;
      do
      {
        v29 = v28;
        v28 += 8;
        *(_QWORD *)(v23 + v29) = *(_QWORD *)&v25[v29];
      }
      while ( v28 < v27 );
    }
    *(_QWORD *)(v3 + 32) += v17;
  }
  else
  {
    sub_CB6200(v3, (unsigned __int8 *)v16, v17);
  }
  v19 = *(_QWORD *)(a2 + 40);
  v20 = v19 + 8LL * *(unsigned int *)(a2 + 48);
  while ( v20 != v19 )
  {
    v19 += 8;
    v21 = sub_CB69B0(v3, 2u);
    sub_30B0EC0(v21, *(_QWORD *)(v19 - 8));
  }
  return v3;
}
