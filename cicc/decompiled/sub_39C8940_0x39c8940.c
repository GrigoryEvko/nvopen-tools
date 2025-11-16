// Function: sub_39C8940
// Address: 0x39c8940
//
void __fastcall sub_39C8940(__int64 a1, __int64 a2, __int64 a3, unsigned __int8 *a4)
{
  __int64 v7; // rdi
  _BYTE *v8; // rax
  __int64 v9; // rdx
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // rdi
  __m128i *v12; // rax
  __m128i *v13; // rdx
  __m128i *v14; // rcx
  unsigned int v15; // r9d
  _QWORD *v16; // r10
  __int64 v17; // rdx
  __int64 v18; // rax
  unsigned int v19; // r9d
  _QWORD *v20; // r10
  _QWORD *v21; // rcx
  _BYTE *v22; // rdi
  __int64 *v23; // rax
  __int64 *v24; // rax
  __int64 v25; // rax
  _BYTE *v26; // rax
  _QWORD *v27; // [rsp+8h] [rbp-B8h]
  _QWORD *v28; // [rsp+8h] [rbp-B8h]
  unsigned int v29; // [rsp+10h] [rbp-B0h]
  _QWORD *v30; // [rsp+10h] [rbp-B0h]
  _QWORD *v31; // [rsp+10h] [rbp-B0h]
  _QWORD *v32; // [rsp+18h] [rbp-A8h]
  unsigned int v33; // [rsp+18h] [rbp-A8h]
  unsigned int v34; // [rsp+20h] [rbp-A0h]
  __m128i *src; // [rsp+30h] [rbp-90h]
  size_t n; // [rsp+38h] [rbp-88h]
  __m128i v37; // [rsp+40h] [rbp-80h] BYREF
  char *v38; // [rsp+50h] [rbp-70h] BYREF
  size_t v39; // [rsp+58h] [rbp-68h]
  _QWORD v40[2]; // [rsp+60h] [rbp-60h] BYREF
  __m128i v41; // [rsp+70h] [rbp-50h] BYREF
  _QWORD v42[8]; // [rsp+80h] [rbp-40h] BYREF

  if ( (unsigned __int8)sub_39C8520((_QWORD *)a1) )
  {
    v7 = *(_QWORD *)(a2 + 8 * (2LL - *(unsigned int *)(a2 + 8)));
    if ( v7 && (v8 = (_BYTE *)sub_161E970(v7)) != 0 )
    {
      v38 = (char *)v40;
      sub_39C7500((__int64 *)&v38, v8, (__int64)&v8[v9]);
    }
    else
    {
      v39 = 0;
      v38 = (char *)v40;
      LOBYTE(v40[0]) = 0;
    }
    sub_39A2AF0(&v41, a1, a4);
    v10 = 15;
    v11 = 15;
    if ( (_QWORD *)v41.m128i_i64[0] != v42 )
      v11 = v42[0];
    if ( v41.m128i_i64[1] + v39 <= v11 )
      goto LABEL_12;
    if ( v38 != (char *)v40 )
      v10 = v40[0];
    if ( v41.m128i_i64[1] + v39 <= v10 )
    {
      v12 = (__m128i *)sub_2241130((unsigned __int64 *)&v38, 0, 0, v41.m128i_i64[0], v41.m128i_u64[1]);
      src = &v37;
      v14 = (__m128i *)v12->m128i_i64[0];
      v13 = v12 + 1;
      if ( (__m128i *)v12->m128i_i64[0] != &v12[1] )
        goto LABEL_13;
    }
    else
    {
LABEL_12:
      v12 = (__m128i *)sub_2241490((unsigned __int64 *)&v41, v38, v39);
      v13 = v12 + 1;
      src = &v37;
      v14 = (__m128i *)v12->m128i_i64[0];
      if ( (__m128i *)v12->m128i_i64[0] != &v12[1] )
      {
LABEL_13:
        src = v14;
        v37.m128i_i64[0] = v12[1].m128i_i64[0];
        goto LABEL_14;
      }
    }
    v37 = _mm_loadu_si128(v12 + 1);
LABEL_14:
    n = v12->m128i_u64[1];
    v12->m128i_i64[0] = (__int64)v13;
    v12->m128i_i64[1] = 0;
    v12[1].m128i_i8[0] = 0;
    if ( (_QWORD *)v41.m128i_i64[0] != v42 )
      j_j___libc_free_0(v41.m128i_u64[0]);
    if ( v38 != (char *)v40 )
      j_j___libc_free_0((unsigned __int64)v38);
    v15 = sub_16D19C0(a1 + 704, (unsigned __int8 *)src, n);
    v16 = (_QWORD *)(*(_QWORD *)(a1 + 704) + 8LL * v15);
    v17 = *v16;
    if ( *v16 )
    {
      if ( v17 != -8 )
      {
LABEL_20:
        *(_QWORD *)(v17 + 8) = a3;
        if ( src != &v37 )
          j_j___libc_free_0((unsigned __int64)src);
        return;
      }
      --*(_DWORD *)(a1 + 720);
    }
    v27 = v16;
    v29 = v15;
    v18 = malloc(n + 17);
    v19 = v29;
    v20 = v27;
    v21 = (_QWORD *)v18;
    if ( !v18 )
    {
      if ( n == -17 )
      {
        v25 = malloc(1u);
        v19 = v29;
        v20 = v27;
        v21 = 0;
        if ( v25 )
        {
          v22 = (_BYTE *)(v25 + 16);
          v21 = (_QWORD *)v25;
          goto LABEL_34;
        }
      }
      v28 = v21;
      v31 = v20;
      v33 = v19;
      sub_16BD1C0("Allocation failed", 1u);
      v19 = v33;
      v20 = v31;
      v21 = v28;
    }
    v22 = v21 + 2;
    if ( n + 1 <= 1 )
    {
LABEL_25:
      v22[n] = 0;
      *v21 = n;
      v21[1] = 0;
      *v20 = v21;
      ++*(_DWORD *)(a1 + 716);
      v23 = (__int64 *)(*(_QWORD *)(a1 + 704) + 8LL * (unsigned int)sub_16D1CD0(a1 + 704, v19));
      v17 = *v23;
      if ( *v23 == -8 || !v17 )
      {
        v24 = v23 + 1;
        do
        {
          do
            v17 = *v24++;
          while ( v17 == -8 );
        }
        while ( !v17 );
      }
      goto LABEL_20;
    }
LABEL_34:
    v30 = v21;
    v32 = v20;
    v34 = v19;
    v26 = memcpy(v22, src, n);
    v21 = v30;
    v20 = v32;
    v19 = v34;
    v22 = v26;
    goto LABEL_25;
  }
}
