// Function: sub_2F2A2A0
// Address: 0x2f2a2a0
//
__int64 __fastcall sub_2F2A2A0(__int64 a1, unsigned int a2, __int64 a3, __int64 a4)
{
  _DWORD *v8; // rax
  __int64 v9; // rdx
  __int64 result; // rax
  __int64 v11; // rax
  __int64 v12; // rdi
  __int64 v13; // rsi
  __int64 v14; // rcx
  __int64 v15; // rdx
  __int64 v16; // rdx
  __int64 i; // rax
  __int64 v18; // rbx
  __int64 v19; // r12
  int v20; // eax
  __int64 v21; // rdi
  __int64 v22; // rdx
  unsigned __int64 v23; // rax
  __int64 v24; // r9
  __int64 v25; // rax
  const __m128i *v26; // rbx
  __m128i *v27; // rax
  unsigned __int64 v28; // rax
  __int64 v29; // r9
  __int64 v30; // r8
  const void *v31; // rsi
  __int64 v32; // r8
  const void *v33; // rsi
  int v34; // [rsp+8h] [rbp-68h]
  __int64 v35; // [rsp+8h] [rbp-68h]
  __int64 v36; // [rsp+8h] [rbp-68h]
  int v37; // [rsp+18h] [rbp-58h] BYREF
  int v38; // [rsp+1Ch] [rbp-54h] BYREF
  __int64 v39; // [rsp+20h] [rbp-50h] BYREF
  int v40; // [rsp+28h] [rbp-48h]
  int v41; // [rsp+2Ch] [rbp-44h]
  char v42; // [rsp+30h] [rbp-40h]

  if ( !*(_QWORD *)(a3 + 64) )
  {
    v8 = *(_DWORD **)a3;
    v9 = *(_QWORD *)a3 + 4LL * *(unsigned int *)(a3 + 8);
    if ( v8 != (_DWORD *)v9 )
    {
      while ( *v8 != a2 )
      {
        if ( (_DWORD *)v9 == ++v8 )
          goto LABEL_3;
      }
      if ( (_DWORD *)v9 != v8 )
        return 1;
    }
LABEL_3:
    if ( !(unsigned __int8)sub_2EBEF70(*(_QWORD *)(a1 + 24), a2) )
      return 0;
    goto LABEL_4;
  }
  v11 = *(_QWORD *)(a3 + 40);
  v12 = a3 + 32;
  if ( !v11 )
    goto LABEL_3;
  v13 = a3 + 32;
  do
  {
    while ( 1 )
    {
      v14 = *(_QWORD *)(v11 + 16);
      v15 = *(_QWORD *)(v11 + 24);
      if ( *(_DWORD *)(v11 + 32) >= a2 )
        break;
      v11 = *(_QWORD *)(v11 + 24);
      if ( !v15 )
        goto LABEL_16;
    }
    v13 = v11;
    v11 = *(_QWORD *)(v11 + 16);
  }
  while ( v14 );
LABEL_16:
  if ( v12 == v13 )
    goto LABEL_3;
  result = 1;
  if ( *(_DWORD *)(v13 + 32) > a2 )
  {
    if ( !(unsigned __int8)sub_2EBEF70(*(_QWORD *)(a1 + 24), a2) )
      return 0;
LABEL_4:
    if ( *(_DWORD *)(a4 + 8) < (unsigned int)qword_5022788 )
    {
      v16 = *(_QWORD *)(a1 + 24);
      for ( i = (a2 & 0x80000000) != 0
              ? *(_QWORD *)(*(_QWORD *)(v16 + 56) + 16LL * (a2 & 0x7FFFFFFF) + 8)
              : *(_QWORD *)(*(_QWORD *)(v16 + 304) + 8LL * a2);
            i && ((*(_BYTE *)(i + 3) & 0x10) != 0 || (*(_BYTE *)(i + 4) & 8) != 0);
            i = *(_QWORD *)(i + 32) )
      {
        ;
      }
      v18 = *(_QWORD *)(i + 16);
      v37 = sub_2E89C70(v18, a2, 0, 0);
      if ( *(_BYTE *)(*(_QWORD *)(v18 + 16) + 4LL) == 1 )
      {
        v19 = *(_QWORD *)(v18 + 32);
        if ( !*(_BYTE *)v19
          && *(int *)(v19 + 8) < 0
          && (*(_BYTE *)(v19 + 3) & 0x10) != 0
          && (*(_WORD *)(v19 + 2) & 0xFF0) != 0 )
        {
          v20 = sub_2E89F40(v18, 0);
          if ( v37 == v20 )
          {
            v22 = *(unsigned int *)(a4 + 8);
            v28 = *(unsigned int *)(a4 + 12);
            v39 = v18;
            v42 = 0;
            v29 = v22 + 1;
            if ( v22 + 1 > v28 )
            {
              v32 = *(_QWORD *)a4;
              v26 = (const __m128i *)&v39;
              v33 = (const void *)(a4 + 16);
              if ( *(_QWORD *)a4 > (unsigned __int64)&v39
                || (v36 = *(_QWORD *)a4, (unsigned __int64)&v39 >= v32 + 24 * v22) )
              {
                sub_C8D5F0(a4, v33, v22 + 1, 0x18u, v32, v29);
                v22 = *(unsigned int *)(a4 + 8);
                v25 = *(_QWORD *)a4;
              }
              else
              {
                sub_C8D5F0(a4, v33, v22 + 1, 0x18u, v32, v29);
                v25 = *(_QWORD *)a4;
                v22 = *(unsigned int *)(a4 + 8);
                v26 = (const __m128i *)((char *)&v39 + *(_QWORD *)a4 - v36);
              }
            }
            else
            {
              v25 = *(_QWORD *)a4;
              v26 = (const __m128i *)&v39;
            }
            goto LABEL_36;
          }
          v21 = *(_QWORD *)(a1 + 8);
          v34 = v20;
          v38 = -1;
          if ( (*(unsigned __int8 (__fastcall **)(__int64, __int64, int *, int *))(*(_QWORD *)v21 + 264LL))(
                 v21,
                 v18,
                 &v37,
                 &v38)
            && v38 == v34 )
          {
            v22 = *(unsigned int *)(a4 + 8);
            v39 = v18;
            v41 = v34;
            v40 = v37;
            v23 = *(unsigned int *)(a4 + 12);
            v24 = v22 + 1;
            v42 = 1;
            if ( v22 + 1 > v23 )
            {
              v30 = *(_QWORD *)a4;
              v26 = (const __m128i *)&v39;
              v31 = (const void *)(a4 + 16);
              if ( *(_QWORD *)a4 > (unsigned __int64)&v39
                || (v35 = *(_QWORD *)a4, (unsigned __int64)&v39 >= v30 + 24 * v22) )
              {
                sub_C8D5F0(a4, v31, v22 + 1, 0x18u, v30, v24);
                v22 = *(unsigned int *)(a4 + 8);
                v25 = *(_QWORD *)a4;
              }
              else
              {
                sub_C8D5F0(a4, v31, v22 + 1, 0x18u, v30, v24);
                v25 = *(_QWORD *)a4;
                v22 = *(unsigned int *)(a4 + 8);
                v26 = (const __m128i *)((char *)&v39 + *(_QWORD *)a4 - v35);
              }
            }
            else
            {
              v25 = *(_QWORD *)a4;
              v26 = (const __m128i *)&v39;
            }
LABEL_36:
            v27 = (__m128i *)(24 * v22 + v25);
            *v27 = _mm_loadu_si128(v26);
            v27[1].m128i_i64[0] = v26[1].m128i_i64[0];
            ++*(_DWORD *)(a4 + 8);
            return sub_2F2A2A0(a1, *(unsigned int *)(v19 + 8), a3, a4);
          }
        }
      }
    }
    return 0;
  }
  return result;
}
