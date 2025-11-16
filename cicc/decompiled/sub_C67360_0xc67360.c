// Function: sub_C67360
// Address: 0xc67360
//
__int64 *__fastcall sub_C67360(__int64 *a1, __int64 a2, _DWORD *a3)
{
  unsigned __int64 v5; // rax
  _QWORD *v6; // rdi
  __m128i *v7; // rax
  __int64 v8; // rsi
  _QWORD *v9; // r9
  __m128i *v10; // rdx
  char v11; // cl
  unsigned int v12; // eax
  __int64 v13; // rdx
  unsigned int v14; // r13d
  __int64 v15; // rax
  __int64 v16; // r8
  __int64 v17; // rdx
  __int64 v18; // r9
  __int64 v19; // rdi
  _BYTE *v20; // rax
  char *v21; // rsi
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rdi
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rdx
  __int64 v29; // rax
  __int64 v30; // [rsp+8h] [rbp-178h]
  __int64 v31; // [rsp+8h] [rbp-178h]
  __int64 *v32; // [rsp+20h] [rbp-160h] BYREF
  unsigned __int64 v33; // [rsp+28h] [rbp-158h]
  __int64 v34; // [rsp+30h] [rbp-150h] BYREF
  _QWORD v35[2]; // [rsp+40h] [rbp-140h] BYREF
  __m128i v36; // [rsp+50h] [rbp-130h] BYREF
  __m128i *v37; // [rsp+60h] [rbp-120h] BYREF
  __int64 v38; // [rsp+68h] [rbp-118h]
  __m128i v39; // [rsp+70h] [rbp-110h] BYREF
  _QWORD *v40; // [rsp+80h] [rbp-100h] BYREF
  __int64 v41; // [rsp+88h] [rbp-F8h]
  _QWORD v42[2]; // [rsp+90h] [rbp-F0h] BYREF
  __int16 v43; // [rsp+A0h] [rbp-E0h]
  char *v44; // [rsp+B0h] [rbp-D0h] BYREF
  __int64 v45; // [rsp+B8h] [rbp-C8h]
  __int64 v46; // [rsp+C0h] [rbp-C0h]
  _BYTE v47[184]; // [rsp+C8h] [rbp-B8h] BYREF

  *a3 = -1;
  v44 = v47;
  v45 = 0;
  v46 = 128;
  sub_CA0F50(&v32, a2);
  v5 = v33;
  if ( v33 > 0x8C )
  {
    sub_22410F0(&v32, 140, 0);
    v5 = v33;
  }
  v37 = &v39;
  sub_C66B70((__int64 *)&v37, v32, (__int64)v32 + v5);
  v40 = v42;
  sub_C66AC0((__int64 *)&v40, "/", (__int64)"");
  v6 = v40;
  v7 = v37;
  v8 = v38;
  v9 = (_QWORD *)((char *)v40 + v41);
  if ( v40 != (_QWORD *)((char *)v40 + v41) )
  {
    do
    {
      v10 = (__m128i *)((char *)v7 + v8);
      v11 = *(_BYTE *)v6;
      if ( &v7->m128i_i8[v8] != (__int8 *)v7 )
      {
        do
        {
          if ( v11 == v7->m128i_i8[0] )
            v7->m128i_i8[0] = 95;
          v7 = (__m128i *)((char *)v7 + 1);
        }
        while ( v10 != v7 );
        v7 = v37;
        v8 = v38;
      }
      v6 = (_QWORD *)((char *)v6 + 1);
    }
    while ( v9 != v6 );
    v6 = v40;
  }
  v35[0] = &v36;
  if ( v7 == &v39 )
  {
    v36 = _mm_load_si128(&v39);
  }
  else
  {
    v35[0] = v7;
    v36.m128i_i64[0] = v39.m128i_i64[0];
  }
  v35[1] = v8;
  v37 = &v39;
  v38 = 0;
  v39.m128i_i8[0] = 0;
  if ( v6 != v42 )
  {
    j_j___libc_free_0(v6, v42[0] + 1LL);
    if ( v37 != &v39 )
      j_j___libc_free_0(v37, v39.m128i_i64[0] + 1);
  }
  v43 = 260;
  v40 = v35;
  v12 = sub_C85AA0(&v40, "dot", 3, a3, &v44, 0);
  v30 = v13;
  v14 = v12;
  if ( v12 )
  {
    v15 = sub_CB72A0(&v40, "dot");
    v16 = v30;
    v17 = *(_QWORD *)(v15 + 32);
    v18 = v15;
    if ( (unsigned __int64)(*(_QWORD *)(v15 + 24) - v17) <= 6 )
    {
      v29 = sub_CB6200(v15, "Error: ", 7);
      v16 = v30;
      v18 = v29;
    }
    else
    {
      *(_DWORD *)v17 = 1869771333;
      *(_WORD *)(v17 + 4) = 14962;
      *(_BYTE *)(v17 + 6) = 32;
      *(_QWORD *)(v15 + 32) += 7LL;
    }
    v31 = v18;
    (*(void (__fastcall **)(_QWORD **, __int64, _QWORD))(*(_QWORD *)v16 + 32LL))(&v40, v16, v14);
    v19 = sub_CB6200(v31, v40, v41);
    v20 = *(_BYTE **)(v19 + 32);
    if ( *(_BYTE **)(v19 + 24) == v20 )
    {
      sub_CB6200(v19, "\n", 1);
    }
    else
    {
      *v20 = 10;
      ++*(_QWORD *)(v19 + 32);
    }
    if ( v40 != v42 )
      j_j___libc_free_0(v40, v42[0] + 1LL);
    *a1 = (__int64)(a1 + 2);
    v21 = (char *)byte_3F871B3;
    sub_C66AC0(a1, byte_3F871B3, (__int64)byte_3F871B3);
  }
  else
  {
    v22 = sub_CB72A0(&v40, "dot");
    v23 = *(_QWORD *)(v22 + 32);
    v24 = v22;
    if ( (unsigned __int64)(*(_QWORD *)(v22 + 24) - v23) <= 8 )
    {
      v24 = sub_CB6200(v22, "Writing '", 9);
    }
    else
    {
      *(_BYTE *)(v23 + 8) = 39;
      *(_QWORD *)v23 = 0x20676E6974697257LL;
      *(_QWORD *)(v22 + 32) += 9LL;
    }
    v25 = sub_CB6200(v24, v44, v45);
    v26 = *(_QWORD *)(v25 + 32);
    if ( (unsigned __int64)(*(_QWORD *)(v25 + 24) - v26) <= 4 )
    {
      sub_CB6200(v25, "'... ", 5);
    }
    else
    {
      *(_DWORD *)v26 = 774778407;
      *(_BYTE *)(v26 + 4) = 32;
      *(_QWORD *)(v25 + 32) += 5LL;
    }
    v21 = v44;
    v27 = v45;
    *a1 = (__int64)(a1 + 2);
    sub_C66AC0(a1, v21, (__int64)&v21[v27]);
  }
  if ( (__m128i *)v35[0] != &v36 )
  {
    v21 = (char *)(v36.m128i_i64[0] + 1);
    j_j___libc_free_0(v35[0], v36.m128i_i64[0] + 1);
  }
  if ( v32 != &v34 )
  {
    v21 = (char *)(v34 + 1);
    j_j___libc_free_0(v32, v34 + 1);
  }
  if ( v44 != v47 )
    _libc_free(v44, v21);
  return a1;
}
