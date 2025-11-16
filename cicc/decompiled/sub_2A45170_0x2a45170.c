// Function: sub_2A45170
// Address: 0x2a45170
//
void __fastcall sub_2A45170(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx
  __int64 v8; // rcx
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rcx
  unsigned int v13; // eax
  __int64 v14; // rax
  __int64 v15; // rax
  unsigned __int64 v16; // rcx
  const __m128i *v17; // r15
  unsigned __int64 v18; // rdx
  __int64 v19; // rax
  unsigned __int64 v20; // r8
  __m128i *v21; // rax
  char *v22; // rax
  unsigned __int8 v23; // dl
  const void *v24; // rsi
  char *v25; // r15
  __int64 v26; // [rsp+0h] [rbp-60h] BYREF
  int v27; // [rsp+8h] [rbp-58h]
  __int64 v28; // [rsp+10h] [rbp-50h]
  __int64 v29; // [rsp+18h] [rbp-48h]
  __int64 v30; // [rsp+20h] [rbp-40h]
  char v31; // [rsp+28h] [rbp-38h]

  v6 = *(_QWORD *)(a2 + 16);
  if ( v6 )
  {
    while ( 1 )
    {
      v22 = *(char **)(v6 + 24);
      v23 = *v22;
      if ( (unsigned __int8)*v22 > 0x1Cu )
        break;
LABEL_9:
      v6 = *(_QWORD *)(v6 + 8);
      if ( !v6 )
        return;
    }
    v27 = 1;
    v28 = 0;
    v30 = 0;
    v31 = 0;
    if ( v23 == 84 )
    {
      v8 = *((_QWORD *)v22 - 1);
      v9 = *((unsigned int *)v22 + 18);
      v27 = 2;
      v10 = *(_QWORD *)(a1 + 16);
      v11 = *(_QWORD *)(v8 + 32 * v9 + 8LL * (unsigned int)((v6 - v8) >> 5));
      if ( !v11 )
      {
LABEL_13:
        v12 = 0;
        v13 = 0;
        goto LABEL_5;
      }
    }
    else
    {
      v11 = *((_QWORD *)v22 + 5);
      v10 = *(_QWORD *)(a1 + 16);
      if ( !v11 )
        goto LABEL_13;
    }
    v12 = (unsigned int)(*(_DWORD *)(v11 + 44) + 1);
    v13 = *(_DWORD *)(v11 + 44) + 1;
LABEL_5:
    if ( v13 < *(_DWORD *)(v10 + 32) )
    {
      v14 = *(_QWORD *)(*(_QWORD *)(v10 + 24) + 8 * v12);
      if ( v14 )
      {
        v15 = *(_QWORD *)(v14 + 72);
        v16 = *(unsigned int *)(a3 + 12);
        v29 = v6;
        v17 = (const __m128i *)&v26;
        v18 = *(_QWORD *)a3;
        v26 = v15;
        v19 = *(unsigned int *)(a3 + 8);
        v20 = v19 + 1;
        if ( v19 + 1 > v16 )
        {
          v24 = (const void *)(a3 + 16);
          if ( v18 > (unsigned __int64)&v26 || (unsigned __int64)&v26 >= v18 + 48 * v19 )
          {
            sub_C8D5F0(a3, v24, v20, 0x30u, v20, a6);
            v18 = *(_QWORD *)a3;
            v19 = *(unsigned int *)(a3 + 8);
          }
          else
          {
            v25 = (char *)&v26 - v18;
            sub_C8D5F0(a3, v24, v20, 0x30u, v20, a6);
            v18 = *(_QWORD *)a3;
            v19 = *(unsigned int *)(a3 + 8);
            v17 = (const __m128i *)&v25[*(_QWORD *)a3];
          }
        }
        v21 = (__m128i *)(v18 + 48 * v19);
        *v21 = _mm_loadu_si128(v17);
        v21[1] = _mm_loadu_si128(v17 + 1);
        v21[2] = _mm_loadu_si128(v17 + 2);
        ++*(_DWORD *)(a3 + 8);
      }
    }
    goto LABEL_9;
  }
}
