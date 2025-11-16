// Function: sub_12186F0
// Address: 0x12186f0
//
__int64 __fastcall sub_12186F0(__int64 a1, __int64 a2, __int64 a3, _BYTE *a4)
{
  __int64 v6; // r12
  int v8; // eax
  __int64 **v9; // rax
  __int64 **v10; // rsi
  unsigned __int64 v11; // r13
  __int64 *v12; // rax
  __int64 v13; // r8
  __int64 v14; // r9
  int v15; // eax
  __int64 v16; // rax
  int v17; // eax
  __int64 **v18; // rsi
  __int64 v19; // rax
  __int64 v20; // rcx
  __int64 v21; // r9
  int v22; // eax
  unsigned __int64 *v23; // rdx
  unsigned __int64 v24; // rax
  int v26; // eax
  __int64 v27; // rax
  unsigned __int64 *v28; // rcx
  unsigned __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // r13
  const __m128i *v33; // rdx
  __int64 v34; // rax
  const __m128i *v35; // rcx
  __int64 v36; // rbx
  __int64 v37; // r13
  __int64 v38; // rdi
  int v39; // eax
  __int64 v40; // [rsp+8h] [rbp-148h]
  __int64 v41; // [rsp+8h] [rbp-148h]
  int v42; // [rsp+10h] [rbp-140h]
  unsigned int v43; // [rsp+18h] [rbp-138h]
  __int64 v44; // [rsp+18h] [rbp-138h]
  unsigned int v46; // [rsp+54h] [rbp-FCh]
  unsigned __int64 v47; // [rsp+68h] [rbp-E8h] BYREF
  __int64 **v48; // [rsp+70h] [rbp-E0h] BYREF
  __int64 v49; // [rsp+78h] [rbp-D8h]
  _QWORD v50[2]; // [rsp+80h] [rbp-D0h] BYREF
  unsigned __int64 v51[4]; // [rsp+90h] [rbp-C0h] BYREF
  char v52; // [rsp+B0h] [rbp-A0h]
  char v53; // [rsp+B1h] [rbp-9Fh]
  __int64 *v54; // [rsp+C0h] [rbp-90h] BYREF
  _BYTE *v55; // [rsp+C8h] [rbp-88h]
  __int64 v56; // [rsp+D0h] [rbp-80h]
  _BYTE v57[120]; // [rsp+D8h] [rbp-78h] BYREF

  v6 = a1 + 176;
  *a4 = 0;
  v8 = sub_1205200(a1 + 176);
  *(_DWORD *)(a1 + 240) = v8;
  if ( v8 == 13 )
    return sub_120AFE0(a1, 13, "expected ')' at end of argument list");
  if ( v8 == 2 )
  {
LABEL_33:
    *(_DWORD *)(a1 + 240) = sub_1205200(v6);
    *a4 = 1;
    return sub_120AFE0(a1, 13, "expected ')' at end of argument list");
  }
  v46 = 0;
  while ( 1 )
  {
    v9 = *(__int64 ***)(a1 + 344);
    v10 = (__int64 **)&v47;
    v11 = *(_QWORD *)(a1 + 232);
    v47 = 0;
    v12 = *v9;
    v53 = 1;
    v52 = 3;
    v54 = v12;
    v55 = v57;
    v56 = 0x800000000LL;
    v51[0] = (unsigned __int64)"expected type";
    if ( (unsigned __int8)sub_12190A0(a1, &v47, v51, 0) )
      break;
    v10 = &v54;
    if ( (unsigned __int8)sub_1218580(a1, &v54, 1u) )
      break;
    if ( *(_BYTE *)(v47 + 8) == 7 )
    {
      v10 = (__int64 **)v11;
      v53 = 1;
      v51[0] = (unsigned __int64)"argument can not have void type";
      v52 = 3;
      sub_11FD800(v6, v11, (__int64)v51, 1);
      break;
    }
    LOBYTE(v50[0]) = 0;
    v48 = (__int64 **)v50;
    v15 = *(_DWORD *)(a1 + 240);
    v49 = 0;
    if ( v15 == 510 )
    {
      sub_2240AE0(&v48, a1 + 248);
      *(_DWORD *)(a1 + 240) = sub_1205200(v6);
    }
    else
    {
      if ( v15 == 504 )
      {
        v10 = (__int64 **)v11;
        v43 = *(_DWORD *)(a1 + 280);
        if ( (unsigned __int8)sub_120EA00(a1, v11, (__int64)"argument", 8, (__int64)"%", 1, v46, v43) )
          goto LABEL_25;
        *(_DWORD *)(a1 + 240) = sub_1205200(v6);
        v46 = v43;
      }
      v16 = *(unsigned int *)(a3 + 8);
      if ( v16 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
      {
        sub_C8D5F0(a3, (const void *)(a3 + 16), v16 + 1, 4u, v13, v14);
        v16 = *(unsigned int *)(a3 + 8);
      }
      *(_DWORD *)(*(_QWORD *)a3 + 4 * v16) = v46;
      ++*(_DWORD *)(a3 + 8);
      ++v46;
    }
    v17 = *(unsigned __int8 *)(v47 + 8);
    if ( v17 == 13 || v17 == 7 )
    {
      v10 = (__int64 **)v11;
      v53 = 1;
      v51[0] = (unsigned __int64)"invalid type for function argument";
      v52 = 3;
      sub_11FD800(v6, v11, (__int64)v51, 1);
LABEL_25:
      if ( v48 != v50 )
      {
        v10 = (__int64 **)(v50[0] + 1LL);
        j_j___libc_free_0(v48, v50[0] + 1LL);
      }
      break;
    }
    v18 = &v54;
    v19 = sub_A7A280(*(__int64 **)v47, (__int64)&v54);
    v20 = *(unsigned int *)(a2 + 8);
    v21 = v19;
    v22 = v20;
    if ( *(_DWORD *)(a2 + 12) <= (unsigned int)v20 )
    {
      v18 = (__int64 **)(a2 + 16);
      v40 = v21;
      v44 = sub_C8D7D0(a2, a2 + 16, 0, 0x38u, v51, v21);
      v27 = 56LL * *(unsigned int *)(a2 + 8);
      v28 = (unsigned __int64 *)(v27 + v44);
      if ( v27 + v44 )
      {
        v29 = v47;
        *v28 = v11;
        v18 = v48;
        v28[1] = v29;
        v30 = v49;
        v28[3] = (unsigned __int64)(v28 + 5);
        v28[2] = v40;
        sub_12060D0((__int64 *)v28 + 3, v18, (__int64)v18 + v30);
        v27 = 56LL * *(unsigned int *)(a2 + 8);
      }
      v31 = *(_QWORD *)a2;
      v32 = *(_QWORD *)a2 + v27;
      if ( *(_QWORD *)a2 != v32 )
      {
        v33 = (const __m128i *)(v31 + 40);
        v34 = v44;
        v18 = (__int64 **)(v44
                         + 56
                         * (((0xDB6DB6DB6DB6DB7LL * ((unsigned __int64)(v32 - v31 - 56) >> 3)) & 0x1FFFFFFFFFFFFFFFLL)
                          + 1));
        do
        {
          if ( v34 )
          {
            *(_QWORD *)v34 = v33[-3].m128i_i64[1];
            *(_QWORD *)(v34 + 8) = v33[-2].m128i_i64[0];
            *(_QWORD *)(v34 + 16) = v33[-2].m128i_i64[1];
            *(_QWORD *)(v34 + 24) = v34 + 40;
            v35 = (const __m128i *)v33[-1].m128i_i64[0];
            if ( v33 == v35 )
            {
              *(__m128i *)(v34 + 40) = _mm_loadu_si128(v33);
            }
            else
            {
              *(_QWORD *)(v34 + 24) = v35;
              *(_QWORD *)(v34 + 40) = v33->m128i_i64[0];
            }
            *(_QWORD *)(v34 + 32) = v33[-1].m128i_i64[1];
            v33[-1].m128i_i64[0] = (__int64)v33;
            v33[-1].m128i_i64[1] = 0;
            v33->m128i_i8[0] = 0;
          }
          v34 += 56;
          v33 = (const __m128i *)((char *)v33 + 56);
        }
        while ( v18 != (__int64 **)v34 );
        v32 = *(_QWORD *)a2 + 56LL * *(unsigned int *)(a2 + 8);
        if ( *(_QWORD *)a2 != v32 )
        {
          v41 = a3;
          v36 = *(_QWORD *)a2 + 56LL * *(unsigned int *)(a2 + 8);
          v37 = *(_QWORD *)a2;
          do
          {
            v36 -= 56;
            v38 = *(_QWORD *)(v36 + 24);
            if ( v38 != v36 + 40 )
            {
              v18 = (__int64 **)(*(_QWORD *)(v36 + 40) + 1LL);
              j_j___libc_free_0(v38, v18);
            }
          }
          while ( v36 != v37 );
          a3 = v41;
          v32 = *(_QWORD *)a2;
        }
      }
      v39 = v51[0];
      if ( a2 + 16 != v32 )
      {
        v42 = v51[0];
        _libc_free(v32, v18);
        v39 = v42;
      }
      ++*(_DWORD *)(a2 + 8);
      *(_DWORD *)(a2 + 12) = v39;
      *(_QWORD *)a2 = v44;
    }
    else
    {
      v23 = (unsigned __int64 *)(*(_QWORD *)a2 + 56 * v20);
      if ( v23 )
      {
        v24 = v47;
        v23[2] = v21;
        *v23 = v11;
        v18 = v48;
        v23[1] = v24;
        v23[3] = (unsigned __int64)(v23 + 5);
        sub_12060D0((__int64 *)v23 + 3, v18, (__int64)v18 + v49);
        v22 = *(_DWORD *)(a2 + 8);
      }
      *(_DWORD *)(a2 + 8) = v22 + 1;
    }
    if ( v48 != v50 )
    {
      v18 = (__int64 **)(v50[0] + 1LL);
      j_j___libc_free_0(v48, v50[0] + 1LL);
    }
    if ( v55 != v57 )
      _libc_free(v55, v18);
    if ( *(_DWORD *)(a1 + 240) != 4 )
      return sub_120AFE0(a1, 13, "expected ')' at end of argument list");
    v26 = sub_1205200(v6);
    *(_DWORD *)(a1 + 240) = v26;
    if ( v26 == 2 )
      goto LABEL_33;
  }
  if ( v55 != v57 )
    _libc_free(v55, v10);
  return 1;
}
