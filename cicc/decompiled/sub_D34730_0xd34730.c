// Function: sub_D34730
// Address: 0xd34730
//
__int64 __fastcall sub_D34730(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 result; // rax
  unsigned int v5; // r13d
  unsigned int i; // r14d
  unsigned __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rdi
  __int64 v11; // rax
  _WORD *v12; // rdx
  __int64 v13; // rax
  __m128i *v14; // rdx
  __int64 v15; // rdi
  __m128i si128; // xmm0
  __int64 v17; // rax
  __int64 v18; // rdx
  unsigned int *v19; // rcx
  unsigned int *v20; // r15
  __int64 v21; // rbx
  __int64 v22; // r14
  _BYTE *v23; // rax
  __int64 v24; // rax
  void *v25; // rdx
  __int64 v26; // rdi
  __int64 v27; // rax
  __int64 v28; // rdx
  unsigned int *v29; // rcx
  unsigned int *v30; // r15
  __int64 v31; // rbx
  __int64 v32; // r14
  _BYTE *v33; // rax
  unsigned __int64 *v34; // [rsp+0h] [rbp-60h]
  unsigned int v36; // [rsp+Ch] [rbp-54h]
  unsigned __int64 v37; // [rsp+10h] [rbp-50h]
  unsigned __int64 *v38; // [rsp+18h] [rbp-48h]
  unsigned int *v39; // [rsp+20h] [rbp-40h]
  unsigned int *v40; // [rsp+20h] [rbp-40h]

  result = *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8);
  v38 = *(unsigned __int64 **)a3;
  v34 = (unsigned __int64 *)result;
  if ( *(_QWORD *)a3 != result )
  {
    v5 = a4 + 2;
    for ( i = 0; ; i = v36 )
    {
      v7 = *v38;
      v37 = v38[1];
      v8 = sub_CB69B0(a2, a4);
      v9 = *(_QWORD *)(v8 + 32);
      v10 = v8;
      if ( (unsigned __int64)(*(_QWORD *)(v8 + 24) - v9) <= 5 )
      {
        v10 = sub_CB6200(v8, "Check ", 6u);
      }
      else
      {
        *(_DWORD *)v9 = 1667590211;
        *(_WORD *)(v9 + 4) = 8299;
        *(_QWORD *)(v8 + 32) += 6LL;
      }
      v36 = i + 1;
      v11 = sub_CB59D0(v10, i);
      v12 = *(_WORD **)(v11 + 32);
      if ( *(_QWORD *)(v11 + 24) - (_QWORD)v12 <= 1u )
      {
        sub_CB6200(v11, (unsigned __int8 *)":\n", 2u);
      }
      else
      {
        *v12 = 2618;
        *(_QWORD *)(v11 + 32) += 2LL;
      }
      v13 = sub_CB69B0(a2, v5);
      v14 = *(__m128i **)(v13 + 32);
      v15 = v13;
      if ( *(_QWORD *)(v13 + 24) - (_QWORD)v14 <= 0x10u )
      {
        v15 = sub_CB6200(v13, "Comparing group (", 0x11u);
      }
      else
      {
        si128 = _mm_load_si128((const __m128i *)&xmmword_3F717C0);
        v14[1].m128i_i8[0] = 40;
        *v14 = si128;
        *(_QWORD *)(v13 + 32) += 17LL;
      }
      v17 = sub_CB5A80(v15, *v38);
      v18 = *(_QWORD *)(v17 + 32);
      if ( (unsigned __int64)(*(_QWORD *)(v17 + 24) - v18) <= 2 )
      {
        sub_CB6200(v17, (unsigned __int8 *)"):\n", 3u);
      }
      else
      {
        *(_BYTE *)(v18 + 2) = 10;
        *(_WORD *)v18 = 14889;
        *(_QWORD *)(v17 + 32) += 3LL;
      }
      v19 = *(unsigned int **)(v7 + 16);
      v20 = v19;
      v39 = &v19[*(unsigned int *)(v7 + 24)];
      if ( v19 != v39 )
      {
        do
        {
          while ( 1 )
          {
            v21 = *v20;
            v22 = sub_CB69B0(a2, v5);
            sub_A69870(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 72 * v21 + 16), (_BYTE *)v22, 0);
            v23 = *(_BYTE **)(v22 + 32);
            if ( *(_BYTE **)(v22 + 24) == v23 )
              break;
            *v23 = 10;
            ++v20;
            ++*(_QWORD *)(v22 + 32);
            if ( v39 == v20 )
              goto LABEL_16;
          }
          ++v20;
          sub_CB6200(v22, (unsigned __int8 *)"\n", 1u);
        }
        while ( v39 != v20 );
      }
LABEL_16:
      v24 = sub_CB69B0(a2, v5);
      v25 = *(void **)(v24 + 32);
      v26 = v24;
      if ( *(_QWORD *)(v24 + 24) - (_QWORD)v25 <= 0xEu )
      {
        v26 = sub_CB6200(v24, "Against group (", 0xFu);
      }
      else
      {
        qmemcpy(v25, "Against group (", 15);
        *(_QWORD *)(v24 + 32) += 15LL;
      }
      v27 = sub_CB5A80(v26, v38[1]);
      v28 = *(_QWORD *)(v27 + 32);
      if ( (unsigned __int64)(*(_QWORD *)(v27 + 24) - v28) <= 2 )
      {
        sub_CB6200(v27, (unsigned __int8 *)"):\n", 3u);
      }
      else
      {
        *(_BYTE *)(v28 + 2) = 10;
        *(_WORD *)v28 = 14889;
        *(_QWORD *)(v27 + 32) += 3LL;
      }
      v29 = *(unsigned int **)(v37 + 16);
      v30 = v29;
      v40 = &v29[*(unsigned int *)(v37 + 24)];
      if ( v40 != v29 )
      {
        do
        {
          while ( 1 )
          {
            v31 = *v30;
            v32 = sub_CB69B0(a2, v5);
            sub_A69870(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 72 * v31 + 16), (_BYTE *)v32, 0);
            v33 = *(_BYTE **)(v32 + 32);
            if ( *(_BYTE **)(v32 + 24) == v33 )
              break;
            *v33 = 10;
            ++v30;
            ++*(_QWORD *)(v32 + 32);
            if ( v40 == v30 )
              goto LABEL_25;
          }
          ++v30;
          sub_CB6200(v32, (unsigned __int8 *)"\n", 1u);
        }
        while ( v40 != v30 );
      }
LABEL_25:
      v38 += 2;
      result = (__int64)v38;
      if ( v34 == v38 )
        break;
    }
  }
  return result;
}
