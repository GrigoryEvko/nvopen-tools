// Function: sub_12D48A0
// Address: 0x12d48a0
//
void __fastcall sub_12D48A0(__m128i *a1, const __m128i *a2, __int64 a3)
{
  const __m128i *i; // r12
  int v5; // eax
  __int64 v6; // rax
  unsigned int v7; // ebx
  int v8; // eax
  __int64 v9; // rax
  unsigned int v10; // eax
  __int64 v11; // r14
  __int64 v12; // r13
  const __m128i *j; // rbx
  __int64 v14; // rax
  int v15; // eax
  __int64 v16; // rax
  int v17; // eax
  __int64 *v18; // [rsp+10h] [rbp-40h]
  unsigned int v19; // [rsp+1Ch] [rbp-34h]

  if ( a1 != a2 )
  {
    for ( i = a1 + 1; a2 != i; v18[1] = v12 )
    {
      while ( 1 )
      {
        v5 = sub_16D1B30(a3, i->m128i_i64[0], i->m128i_i64[1]);
        if ( v5 == -1 || (v6 = *(_QWORD *)a3 + 8LL * v5, v6 == *(_QWORD *)a3 + 8LL * *(unsigned int *)(a3 + 8)) )
          v7 = 0;
        else
          v7 = *(_DWORD *)(*(_QWORD *)v6 + 8LL);
        v8 = sub_16D1B30(a3, a1->m128i_i64[0], a1->m128i_i64[1]);
        if ( v8 == -1 || (v9 = *(_QWORD *)a3 + 8LL * v8, v9 == *(_QWORD *)a3 + 8LL * *(unsigned int *)(a3 + 8)) )
          v10 = 0;
        else
          v10 = *(_DWORD *)(*(_QWORD *)v9 + 8LL);
        v11 = i->m128i_i64[0];
        v12 = i->m128i_i64[1];
        if ( v10 >= v7 )
          break;
        if ( a1 != i )
          memmove(&a1[1], a1, (char *)i - (char *)a1);
        ++i;
        a1->m128i_i64[0] = v11;
        a1->m128i_i64[1] = v12;
        if ( a2 == i )
          return;
      }
      for ( j = i; ; j[1] = _mm_loadu_si128(j) )
      {
        v18 = (__int64 *)j;
        v17 = sub_16D1B30(a3, v11, v12);
        if ( v17 == -1 || (v14 = *(_QWORD *)a3 + 8LL * v17, v14 == *(_QWORD *)a3 + 8LL * *(unsigned int *)(a3 + 8)) )
          v19 = 0;
        else
          v19 = *(_DWORD *)(*(_QWORD *)v14 + 8LL);
        v15 = sub_16D1B30(a3, j[-1].m128i_i64[0], j[-1].m128i_i64[1]);
        if ( v15 == -1 )
          break;
        v16 = *(_QWORD *)a3 + 8LL * v15;
        if ( v16 == *(_QWORD *)a3 + 8LL * *(unsigned int *)(a3 + 8) )
          break;
        --j;
        if ( *(_DWORD *)(*(_QWORD *)v16 + 8LL) >= v19 )
          goto LABEL_24;
LABEL_20:
        ;
      }
      --j;
      if ( v19 )
        goto LABEL_20;
LABEL_24:
      ++i;
      *v18 = v11;
    }
  }
}
