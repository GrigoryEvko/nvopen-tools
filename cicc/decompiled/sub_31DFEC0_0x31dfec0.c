// Function: sub_31DFEC0
// Address: 0x31dfec0
//
void __fastcall sub_31DFEC0(__int64 a1, const void *a2, __int64 *a3, __int64 a4)
{
  __int64 v5; // rax
  __int64 *v6; // rbx
  unsigned __int64 v7; // rax
  unsigned __int8 *v8; // rax
  __int64 v9; // r12
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // r10
  __int64 v13; // rax
  unsigned __int64 v14; // rcx
  __m128i *v15; // rdx
  const __m128i *v16; // r13
  unsigned __int64 v17; // r11
  __m128i *v18; // rax
  __int64 v19; // rax
  __int64 v20; // r13
  unsigned int v21; // edx
  unsigned __int64 *v22; // rdx
  __m128i *v23; // r12
  __int64 v24; // rdx
  __m128i *v25; // r13
  char *v26; // r13
  __int64 v27; // [rsp+8h] [rbp-78h]
  int v28; // [rsp+20h] [rbp-60h]
  __int64 v29; // [rsp+20h] [rbp-60h]
  __int64 *v30; // [rsp+28h] [rbp-58h]
  __int64 v31; // [rsp+30h] [rbp-50h] BYREF
  const __m128i *v32; // [rsp+38h] [rbp-48h]
  __m128i *v33; // [rsp+40h] [rbp-40h]

  if ( *(_BYTE *)a3 != 9 )
    return;
  v5 = 32LL * (*((_DWORD *)a3 + 1) & 0x7FFFFFF);
  if ( (*((_BYTE *)a3 + 7) & 0x40) != 0 )
  {
    v6 = (__int64 *)*(a3 - 1);
    v30 = &v6[(unsigned __int64)v5 / 8];
  }
  else
  {
    v30 = a3;
    v6 = &a3[v5 / 0xFFFFFFFFFFFFFFF8LL];
  }
  if ( v6 != v30 )
  {
    while ( 1 )
    {
      v9 = *v6;
      if ( sub_AC30F0(*(_QWORD *)(*v6 + 32 * (1LL - (*(_DWORD *)(*v6 + 4) & 0x7FFFFFF)))) )
        goto LABEL_22;
      v12 = *(_QWORD *)(v9 - 32LL * (*(_DWORD *)(v9 + 4) & 0x7FFFFFF));
      if ( *(_BYTE *)v12 == 17 )
        break;
LABEL_13:
      v6 += 4;
      if ( v30 == v6 )
        goto LABEL_22;
    }
    v13 = *(unsigned int *)(a4 + 8);
    v14 = *(unsigned int *)(a4 + 12);
    LODWORD(v31) = 0;
    v15 = *(__m128i **)a4;
    v16 = (const __m128i *)&v31;
    v32 = 0;
    v17 = v13 + 1;
    v33 = 0;
    if ( v13 + 1 <= v14 )
    {
LABEL_17:
      v18 = (__m128i *)((char *)v15 + 24 * v13);
      *v18 = _mm_loadu_si128(v16);
      v18[1].m128i_i64[0] = v16[1].m128i_i64[0];
      v19 = (unsigned int)(*(_DWORD *)(a4 + 8) + 1);
      *(_DWORD *)(a4 + 8) = v19;
      v20 = *(_QWORD *)a4 + 24 * v19 - 24;
      if ( *(_DWORD *)(v12 + 32) <= 0x40u )
      {
        v7 = *(_QWORD *)(v12 + 24);
        if ( v7 > 0xFFFF )
          LODWORD(v7) = 0xFFFF;
      }
      else
      {
        v27 = v12;
        v28 = *(_DWORD *)(v12 + 32);
        v21 = v28 - sub_C444A0(v12 + 24);
        LODWORD(v7) = 0xFFFF;
        if ( v21 <= 0x40 )
        {
          LODWORD(v7) = 0xFFFF;
          v22 = *(unsigned __int64 **)(v27 + 24);
          if ( *v22 <= 0xFFFF )
            v7 = *v22;
        }
      }
      *(_DWORD *)v20 = v7;
      *(_QWORD *)(v20 + 8) = *(_QWORD *)(v9 + 32 * (1LL - (*(_DWORD *)(v9 + 4) & 0x7FFFFFF)));
      if ( !sub_AC30F0(*(_QWORD *)(v9 + 32 * (2LL - (*(_DWORD *)(v9 + 4) & 0x7FFFFFF)))) )
      {
        if ( *(_DWORD *)(*(_QWORD *)(a1 + 200) + 556LL) == 19 )
          sub_C64ED0("associated data of XXStructor list is not yet supported on AIX", 1u);
        v8 = sub_BD3990(*(unsigned __int8 **)(v9 + 32 * (2LL - (*(_DWORD *)(v9 + 4) & 0x7FFFFFF))), (__int64)a2);
        if ( *v8 >= 4u )
          v8 = 0;
        *(_QWORD *)(v20 + 16) = v8;
      }
      goto LABEL_13;
    }
    a2 = (const void *)(a4 + 16);
    if ( v15 > (__m128i *)&v31 )
    {
      v29 = v12;
    }
    else
    {
      v29 = v12;
      if ( &v31 < &v15->m128i_i64[3 * v13] )
      {
        v26 = (char *)((char *)&v31 - (char *)v15);
        sub_C8D5F0(a4, a2, v17, 0x18u, v10, v11);
        v15 = *(__m128i **)a4;
        v13 = *(unsigned int *)(a4 + 8);
        v12 = v29;
        v16 = (const __m128i *)&v26[*(_QWORD *)a4];
        goto LABEL_17;
      }
    }
    sub_C8D5F0(a4, a2, v17, 0x18u, v10, v11);
    v15 = *(__m128i **)a4;
    v13 = *(unsigned int *)(a4 + 8);
    v12 = v29;
    goto LABEL_17;
  }
LABEL_22:
  v23 = *(__m128i **)a4;
  v24 = 24LL * *(unsigned int *)(a4 + 8);
  v25 = (__m128i *)(*(_QWORD *)a4 + v24);
  sub_31DFDF0(&v31, *(__m128i **)a4, 0xAAAAAAAAAAAAAAABLL * (v24 >> 3));
  if ( v33 )
    sub_31D7880(v23, v25, v33, v32);
  else
    sub_31D6340(v23->m128i_i8, v25->m128i_i8);
  j_j___libc_free_0((unsigned __int64)v33);
}
