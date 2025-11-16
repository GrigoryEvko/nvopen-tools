// Function: sub_3737010
// Address: 0x3737010
//
void __fastcall sub_3737010(__int64 a1, __int64 a2, unsigned __int8 *a3)
{
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  unsigned __int8 v8; // al
  __int64 v9; // r12
  __int64 v10; // rdi
  _BYTE *v11; // rax
  __int64 v12; // rdx
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // rdi
  __m128i *v15; // rax
  __m128i *v16; // rcx
  __m128i *v17; // rdx
  __m128i *v18; // r9
  int v19; // eax
  unsigned int v20; // r10d
  _QWORD *v21; // r11
  __int64 v22; // rax
  unsigned int v23; // r10d
  _QWORD *v24; // r11
  _QWORD *v25; // rcx
  __m128i *v26; // rsi
  _QWORD *v27; // [rsp+0h] [rbp-C0h]
  _QWORD *v28; // [rsp+0h] [rbp-C0h]
  unsigned int v29; // [rsp+8h] [rbp-B8h]
  _QWORD *v30; // [rsp+8h] [rbp-B8h]
  __m128i *src; // [rsp+10h] [rbp-B0h]
  unsigned int srca; // [rsp+10h] [rbp-B0h]
  __m128i *v33; // [rsp+20h] [rbp-A0h]
  __int64 n; // [rsp+28h] [rbp-98h]
  __m128i v35; // [rsp+30h] [rbp-90h] BYREF
  __m128i v36; // [rsp+40h] [rbp-80h] BYREF
  _QWORD v37[2]; // [rsp+50h] [rbp-70h] BYREF
  __m128i *v38; // [rsp+60h] [rbp-60h] BYREF
  size_t v39; // [rsp+68h] [rbp-58h]
  __m128i v40; // [rsp+70h] [rbp-50h] BYREF
  __int64 v41; // [rsp+80h] [rbp-40h]

  if ( sub_37365C0((_QWORD *)a1) )
  {
    v8 = *(_BYTE *)(a2 - 16);
    if ( (v8 & 2) != 0 )
      v9 = *(_QWORD *)(a2 - 32);
    else
      v9 = a2 - 16 - 8LL * ((v8 >> 2) & 0xF);
    v10 = *(_QWORD *)(v9 + 16);
    if ( v10 && (v11 = (_BYTE *)sub_B91420(v10)) != 0 )
    {
      v38 = &v40;
      sub_3735130((__int64 *)&v38, v11, (__int64)&v11[v12]);
    }
    else
    {
      v39 = 0;
      v38 = &v40;
      v40.m128i_i8[0] = 0;
    }
    sub_3248550(&v36, (char *)a1, a3, v5, v6, v7);
    v13 = 15;
    v14 = 15;
    if ( (_QWORD *)v36.m128i_i64[0] != v37 )
      v14 = v37[0];
    if ( v36.m128i_i64[1] + v39 <= v14 )
      goto LABEL_13;
    if ( v38 != &v40 )
      v13 = v40.m128i_i64[0];
    if ( v36.m128i_i64[1] + v39 <= v13 )
    {
      v15 = (__m128i *)sub_2241130((unsigned __int64 *)&v38, 0, 0, v36.m128i_i64[0], v36.m128i_u64[1]);
      v33 = &v35;
      v16 = (__m128i *)v15->m128i_i64[0];
      v17 = v15 + 1;
      if ( (__m128i *)v15->m128i_i64[0] != &v15[1] )
        goto LABEL_14;
    }
    else
    {
LABEL_13:
      v15 = (__m128i *)sub_2241490((unsigned __int64 *)&v36, v38->m128i_i8, v39);
      v33 = &v35;
      v16 = (__m128i *)v15->m128i_i64[0];
      v17 = v15 + 1;
      if ( (__m128i *)v15->m128i_i64[0] != &v15[1] )
      {
LABEL_14:
        v33 = v16;
        v35.m128i_i64[0] = v15[1].m128i_i64[0];
        goto LABEL_15;
      }
    }
    v35 = _mm_loadu_si128(v15 + 1);
LABEL_15:
    n = v15->m128i_i64[1];
    v15->m128i_i64[0] = (__int64)v17;
    v15->m128i_i64[1] = 0;
    v15[1].m128i_i8[0] = 0;
    if ( (_QWORD *)v36.m128i_i64[0] != v37 )
      j_j___libc_free_0(v36.m128i_u64[0]);
    if ( v38 != &v40 )
      j_j___libc_free_0((unsigned __int64)v38);
    v18 = v33;
    v38 = &v40;
    if ( v33 == &v35 )
    {
      v18 = &v40;
      v40 = _mm_load_si128(&v35);
    }
    else
    {
      v38 = v33;
      v40.m128i_i64[0] = v35.m128i_i64[0];
    }
    v36.m128i_i64[0] = (__int64)v18;
    src = v18;
    v39 = n;
    v35.m128i_i8[0] = 0;
    v41 = a1 + 8;
    v36.m128i_i64[1] = n;
    v19 = sub_C92610();
    v20 = sub_C92740(a1 + 448, (const void *)v36.m128i_i64[0], v36.m128i_u64[1], v19);
    v21 = (_QWORD *)(*(_QWORD *)(a1 + 448) + 8LL * v20);
    if ( *v21 )
    {
      if ( *v21 != -8 )
      {
LABEL_23:
        if ( v38 != &v40 )
          j_j___libc_free_0((unsigned __int64)v38);
        return;
      }
      --*(_DWORD *)(a1 + 464);
    }
    v27 = v21;
    v29 = v20;
    v22 = sub_C7D670(n + 17, 8);
    v23 = v29;
    v24 = v27;
    v25 = (_QWORD *)v22;
    if ( n )
    {
      v26 = src;
      v30 = v27;
      srca = v23;
      v28 = (_QWORD *)v22;
      memcpy((void *)(v22 + 16), v26, n);
      v23 = srca;
      v24 = v30;
      v25 = v28;
    }
    *((_BYTE *)v25 + n + 16) = 0;
    *v25 = n;
    v25[1] = a1 + 8;
    *v24 = v25;
    ++*(_DWORD *)(a1 + 460);
    sub_C929D0((__int64 *)(a1 + 448), v23);
    goto LABEL_23;
  }
}
