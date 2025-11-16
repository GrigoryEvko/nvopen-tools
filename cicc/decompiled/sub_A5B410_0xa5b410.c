// Function: sub_A5B410
// Address: 0xa5b410
//
__int64 __fastcall sub_A5B410(__int64 *a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // r8
  unsigned int v4; // edx
  __int64 v6; // r13
  __int64 v7; // rcx
  __int64 v8; // rdi
  __m128i *v9; // rdx
  __m128i si128; // xmm0
  __int64 v11; // rdi
  __int64 v12; // rdx
  __int64 v13; // r12
  __int64 v14; // r15
  __int64 v15; // r13
  __int64 v16; // r14
  __int64 v17; // rdi
  void *v18; // rdx
  __int64 v19; // rdi
  __int64 v20; // rdi
  _DWORD *v21; // rdx
  unsigned int v22; // r15d
  int v23; // r13d
  __int64 v24; // rdx
  __int64 v25; // rdi
  _WORD *v26; // rdx
  int v27; // edi
  __int64 i; // [rsp-40h] [rbp-40h]

  result = *((unsigned int *)a1 + 88);
  v3 = a1[42];
  if ( (_DWORD)result )
  {
    v4 = (result - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v6 = v3 + 56LL * v4;
    v7 = *(_QWORD *)v6;
    if ( a2 == *(_QWORD *)v6 )
    {
LABEL_3:
      result = v3 + 56 * result;
      if ( v6 != result )
      {
        v8 = *a1;
        v9 = *(__m128i **)(*a1 + 32);
        if ( *(_QWORD *)(*a1 + 24) - (_QWORD)v9 <= 0x1Au )
        {
          sub_CB6200(v8, "\n; uselistorder directives\n", 27);
        }
        else
        {
          si128 = _mm_load_si128((const __m128i *)&xmmword_3F24AF0);
          qmemcpy(&v9[1], "directives\n", 11);
          *v9 = si128;
          *(_QWORD *)(v8 + 32) += 27LL;
        }
        v13 = *(_QWORD *)(v6 + 40);
        v14 = 32LL * *(unsigned int *)(v6 + 48);
        result = v13 + v14;
        for ( i = v13 + v14; i != v13; v13 += 32 )
        {
          v15 = *(_QWORD *)v13;
          v16 = *(_QWORD *)(a1[4] + 16);
          if ( v16 )
            sub_904010(*a1, "  ");
          v17 = *a1;
          v18 = *(void **)(*a1 + 32);
          if ( *(_QWORD *)(*a1 + 24) - (_QWORD)v18 <= 0xBu )
          {
            sub_CB6200(v17, "uselistorder", 12);
          }
          else
          {
            qmemcpy(v18, "uselistorder", 12);
            *(_QWORD *)(v17 + 32) += 12LL;
          }
          v19 = *a1;
          if ( v16 || *(_BYTE *)v15 != 23 )
          {
            sub_904010(v19, " ");
            sub_A5B360(a1, v15, 1);
          }
          else
          {
            sub_904010(v19, "_bb ");
            sub_A5B360(a1, *(_QWORD *)(v15 + 72), 0);
            sub_904010(*a1, ", ");
            sub_A5B360(a1, v15, 0);
          }
          v20 = *a1;
          v21 = *(_DWORD **)(*a1 + 32);
          if ( *(_QWORD *)(*a1 + 24) - (_QWORD)v21 <= 3u )
          {
            sub_CB6200(v20, ", { ", 4);
          }
          else
          {
            *v21 = 544940076;
            *(_QWORD *)(v20 + 32) += 4LL;
          }
          v22 = 1;
          sub_CB59D0(*a1, **(unsigned int **)(v13 + 8));
          v23 = (__int64)(*(_QWORD *)(v13 + 16) - *(_QWORD *)(v13 + 8)) >> 2;
          if ( v23 != 1 )
          {
            do
            {
              v25 = *a1;
              v26 = *(_WORD **)(*a1 + 32);
              if ( *(_QWORD *)(*a1 + 24) - (_QWORD)v26 > 1u )
              {
                *v26 = 8236;
                *(_QWORD *)(v25 + 32) += 2LL;
              }
              else
              {
                v25 = sub_CB6200(v25, ", ", 2);
              }
              v24 = v22++;
              sub_CB59D0(v25, *(unsigned int *)(*(_QWORD *)(v13 + 8) + 4 * v24));
            }
            while ( v23 != v22 );
          }
          v11 = *a1;
          v12 = *(_QWORD *)(*a1 + 32);
          if ( (unsigned __int64)(*(_QWORD *)(*a1 + 24) - v12) <= 2 )
          {
            result = sub_CB6200(v11, " }\n", 3);
          }
          else
          {
            result = 32032;
            *(_BYTE *)(v12 + 2) = 10;
            *(_WORD *)v12 = 32032;
            *(_QWORD *)(v11 + 32) += 3LL;
          }
        }
      }
    }
    else
    {
      v27 = 1;
      while ( v7 != -4096 )
      {
        v4 = (result - 1) & (v27 + v4);
        v6 = v3 + 56LL * v4;
        v7 = *(_QWORD *)v6;
        if ( a2 == *(_QWORD *)v6 )
          goto LABEL_3;
        ++v27;
      }
    }
  }
  return result;
}
