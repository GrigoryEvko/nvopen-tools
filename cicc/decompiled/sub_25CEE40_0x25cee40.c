// Function: sub_25CEE40
// Address: 0x25cee40
//
__int64 __fastcall sub_25CEE40(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  unsigned int v7; // r15d
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rcx
  _BYTE *v12; // rsi
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r9
  unsigned __int64 v16; // rbx
  void *v17; // r12
  size_t v18; // r14
  unsigned __int64 v19; // r13
  size_t v20; // r15
  size_t v21; // rdx
  int v22; // eax
  __int64 v23; // r15
  size_t v24; // r15
  size_t v25; // rdx
  int v26; // eax
  __int64 v27; // r8
  __m128i *v28; // rax
  _QWORD *v29; // rsi
  __m128i *v30; // rbx
  size_t v31; // r15
  __int64 v32; // rax
  _QWORD *v33; // rdx
  _QWORD *v34; // r8
  unsigned int v35; // edi
  unsigned __int64 v36; // rdi
  size_t v37; // rbx
  size_t v38; // rdx
  int v39; // eax
  unsigned int v40; // edi
  __int64 v41; // [rsp+0h] [rbp-D0h]
  unsigned __int64 v42; // [rsp+0h] [rbp-D0h]
  _QWORD *v43; // [rsp+0h] [rbp-D0h]
  unsigned __int64 v44; // [rsp+10h] [rbp-C0h]
  _QWORD *v46; // [rsp+28h] [rbp-A8h]
  __int64 v47; // [rsp+38h] [rbp-98h]
  _QWORD v48[2]; // [rsp+60h] [rbp-70h] BYREF
  __int64 v49; // [rsp+70h] [rbp-60h]
  __int64 v50; // [rsp+78h] [rbp-58h]
  void *s2; // [rsp+80h] [rbp-50h] BYREF
  size_t n; // [rsp+88h] [rbp-48h]
  __m128i v53[4]; // [rsp+90h] [rbp-40h] BYREF

  if ( *(_WORD *)a1 != 7 )
  {
    v7 = 0;
    sub_C6AFF0(&a7, (__int64)"expected object", 15);
    return v7;
  }
  sub_25CD350((_QWORD *)a2[2]);
  a2[2] = 0;
  a2[3] = a2 + 1;
  a2[4] = a2 + 1;
  a2[5] = 0;
  v46 = a2 + 1;
  if ( !*(_DWORD *)(a1 + 24) )
    return 1;
  v9 = *(unsigned int *)(a1 + 32);
  v10 = *(_QWORD *)(a1 + 16);
  v48[0] = a1 + 8;
  v11 = *(_QWORD *)(a1 + 8);
  v49 = v10;
  v48[1] = v11;
  v50 = v10 + (v9 << 6);
  sub_C6B5D0((__int64)v48);
  v47 = v49;
  v44 = *(_QWORD *)(a1 + 16) + ((unsigned __int64)*(unsigned int *)(a1 + 32) << 6);
  if ( v49 == v44 )
    return 1;
  while ( 2 )
  {
    v12 = *(_BYTE **)(v47 + 8);
    v13 = (__int64)&v12[*(_QWORD *)(v47 + 16)];
    s2 = v53;
    sub_25CCF50((__int64 *)&s2, v12, v13);
    v16 = a2[2];
    if ( !v16 )
    {
      v19 = (unsigned __int64)v46;
      goto LABEL_27;
    }
    v17 = s2;
    v18 = n;
    v19 = (unsigned __int64)v46;
    do
    {
      while ( 1 )
      {
        v20 = *(_QWORD *)(v16 + 40);
        v21 = v18;
        if ( v20 <= v18 )
          v21 = *(_QWORD *)(v16 + 40);
        if ( v21 )
        {
          v22 = memcmp(*(const void **)(v16 + 32), v17, v21);
          if ( v22 )
            break;
        }
        v23 = v20 - v18;
        if ( v23 >= 0x80000000LL )
          goto LABEL_17;
        if ( v23 > (__int64)0xFFFFFFFF7FFFFFFFLL )
        {
          v22 = v23;
          break;
        }
LABEL_8:
        v16 = *(_QWORD *)(v16 + 24);
        if ( !v16 )
          goto LABEL_18;
      }
      if ( v22 < 0 )
        goto LABEL_8;
LABEL_17:
      v19 = v16;
      v16 = *(_QWORD *)(v16 + 16);
    }
    while ( v16 );
LABEL_18:
    if ( v46 == (_QWORD *)v19 )
      goto LABEL_27;
    v24 = *(_QWORD *)(v19 + 40);
    v25 = v18;
    if ( v24 <= v18 )
      v25 = *(_QWORD *)(v19 + 40);
    if ( v25 && (v26 = memcmp(v17, *(const void **)(v19 + 32), v25)) != 0 )
    {
LABEL_26:
      if ( v26 < 0 )
        goto LABEL_27;
    }
    else
    {
      v27 = v18 - v24;
      if ( (__int64)(v18 - v24) < 0x80000000LL )
      {
        if ( v27 > (__int64)0xFFFFFFFF7FFFFFFFLL )
        {
          v26 = v18 - v24;
          goto LABEL_26;
        }
LABEL_27:
        v28 = (__m128i *)sub_22077B0(0x58u);
        v29 = (_QWORD *)v19;
        v30 = v28 + 3;
        v19 = (unsigned __int64)v28;
        v28[2].m128i_i64[0] = (__int64)v28[3].m128i_i64;
        if ( s2 == v53 )
        {
          v28[3] = _mm_load_si128(v53);
        }
        else
        {
          v28[2].m128i_i64[0] = (__int64)s2;
          v28[3].m128i_i64[0] = v53[0].m128i_i64[0];
        }
        v31 = n;
        v28[4].m128i_i64[0] = 0;
        v28[4].m128i_i64[1] = 0;
        v28[2].m128i_i64[1] = v31;
        v28[5].m128i_i64[0] = 0;
        s2 = v53;
        n = 0;
        v53[0].m128i_i8[0] = 0;
        v32 = sub_25CEC10(a2, v29, (__int64)v28[2].m128i_i64);
        v34 = v33;
        if ( v33 )
        {
          if ( v32 || v46 == v33 )
          {
LABEL_32:
            LOBYTE(v35) = 1;
            goto LABEL_33;
          }
          v38 = v33[5];
          v37 = v38;
          if ( v31 <= v38 )
            v38 = v31;
          if ( v38
            && (v43 = v34,
                v39 = memcmp(*(const void **)(v19 + 32), (const void *)v34[4], v38),
                v34 = v43,
                (v40 = v39) != 0) )
          {
LABEL_51:
            v35 = v40 >> 31;
          }
          else
          {
            LOBYTE(v35) = 0;
            if ( (__int64)(v31 - v37) < 0x80000000LL )
            {
              if ( (__int64)(v31 - v37) <= (__int64)0xFFFFFFFF7FFFFFFFLL )
                goto LABEL_32;
              v40 = v31 - v37;
              goto LABEL_51;
            }
          }
LABEL_33:
          sub_220F040(v35, v19, v34, v46);
          ++a2[5];
        }
        else
        {
          v36 = *(_QWORD *)(v19 + 32);
          if ( v30 != (__m128i *)v36 )
          {
            v41 = v32;
            j_j___libc_free_0(v36);
            v32 = v41;
          }
          v42 = v32;
          j_j___libc_free_0(v19);
          v19 = v42;
        }
      }
    }
    v7 = sub_25CE440(v47 + 24, v19 + 64, v25, v14, v27, v15, (__int64)&a7);
    if ( s2 != v53 )
      j_j___libc_free_0((unsigned __int64)s2);
    if ( (_BYTE)v7 )
    {
      v49 += 64;
      sub_C6B5D0((__int64)v48);
      v47 = v49;
      if ( v44 == v49 )
        return 1;
      continue;
    }
    return v7;
  }
}
