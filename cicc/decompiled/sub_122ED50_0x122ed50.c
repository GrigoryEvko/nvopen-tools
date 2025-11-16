// Function: sub_122ED50
// Address: 0x122ed50
//
__int64 __fastcall sub_122ED50(__int64 a1, __int64 a2, __int64 *a3, unsigned __int8 a4, unsigned __int8 a5)
{
  __int64 v7; // r14
  __int64 *v8; // rax
  __int64 *v9; // rsi
  __int64 *v10; // rax
  const char **v11; // rsi
  __int64 v12; // rax
  __int64 v13; // r8
  __int64 v14; // r9
  unsigned __int64 v15; // rcx
  __int64 v16; // rax
  unsigned __int64 v17; // rdx
  __int64 v18; // rcx
  const __m128i *v19; // rdx
  __m128i *v20; // rax
  char *v21; // rdi
  unsigned int v22; // r12d
  __int64 v24; // r12
  __int64 v25; // rdi
  unsigned __int64 v26; // rsi
  __int64 v27; // rdi
  unsigned __int64 v28; // rsi
  unsigned __int64 v29; // rsi
  __int64 *v33; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v34; // [rsp+28h] [rbp-B8h] BYREF
  _QWORD v35[4]; // [rsp+30h] [rbp-B0h] BYREF
  const char *v36; // [rsp+50h] [rbp-90h] BYREF
  char *v37; // [rsp+58h] [rbp-88h]
  const char *v38; // [rsp+60h] [rbp-80h]
  char v39; // [rsp+68h] [rbp-78h] BYREF
  __int16 v40; // [rsp+70h] [rbp-70h]

  if ( *(_DWORD *)(a1 + 240) == 13 )
  {
LABEL_25:
    v22 = a5;
    v25 = a1 + 176;
    LOBYTE(v22) = a4 & a5;
    if ( (a4 & a5) != 0 )
    {
      v26 = *(_QWORD *)(a1 + 232);
      v36 = "expected '...' at end of argument list for musttail call in varargs function";
      v40 = 259;
      sub_11FD800(v25, v26, (__int64)&v36, 1);
    }
    else
    {
      *(_DWORD *)(a1 + 240) = sub_1205200(v25);
    }
    return v22;
  }
  while ( 1 )
  {
    if ( *(_DWORD *)(a2 + 8) && (unsigned __int8)sub_120AFE0(a1, 4, "expected ',' in argument list") )
      return 1;
    if ( *(_DWORD *)(a1 + 240) == 2 )
      break;
    v33 = 0;
    v36 = "expected type";
    v7 = *(_QWORD *)(a1 + 232);
    v40 = 259;
    if ( (unsigned __int8)sub_12190A0(a1, &v33, (int *)&v36, 0) )
      return 1;
    v8 = **(__int64 ***)(a1 + 344);
    v37 = &v39;
    v36 = (const char *)v8;
    v38 = (const char *)0x800000000LL;
    if ( *((_BYTE *)v33 + 8) == 9 )
    {
      v9 = &v34;
      if ( (unsigned __int8)sub_12255B0((__int64 **)a1, &v34, a3) )
        goto LABEL_18;
    }
    else
    {
      v9 = (__int64 *)&v36;
      if ( (unsigned __int8)sub_1218580(a1, (__int64 **)&v36, 1u)
        || (v9 = v33, (unsigned __int8)sub_1224B80((__int64 **)a1, (__int64)v33, &v34, a3)) )
      {
LABEL_18:
        if ( v37 != &v39 )
          _libc_free(v37, v9);
        return 1;
      }
    }
    v10 = (__int64 *)sub_BD5C60(v34);
    v11 = &v36;
    v12 = sub_A7A280(v10, (__int64)&v36);
    v15 = *(unsigned int *)(a2 + 12);
    v35[0] = v7;
    v35[2] = v12;
    v16 = *(unsigned int *)(a2 + 8);
    v35[1] = v34;
    v17 = v16 + 1;
    if ( v16 + 1 > v15 )
    {
      v24 = *(_QWORD *)a2;
      v11 = (const char **)(a2 + 16);
      if ( *(_QWORD *)a2 <= (unsigned __int64)v35 && (unsigned __int64)v35 < v24 + 24 * v16 )
      {
        sub_C8D5F0(a2, v11, v17, 0x18u, v13, v14);
        v18 = *(_QWORD *)a2;
        v16 = *(unsigned int *)(a2 + 8);
        v19 = (const __m128i *)((char *)v35 + *(_QWORD *)a2 - v24);
      }
      else
      {
        sub_C8D5F0(a2, v11, v17, 0x18u, v13, v14);
        v18 = *(_QWORD *)a2;
        v16 = *(unsigned int *)(a2 + 8);
        v19 = (const __m128i *)v35;
      }
    }
    else
    {
      v18 = *(_QWORD *)a2;
      v19 = (const __m128i *)v35;
    }
    v20 = (__m128i *)(v18 + 24 * v16);
    *v20 = _mm_loadu_si128(v19);
    v20[1].m128i_i64[0] = v19[1].m128i_i64[0];
    v21 = v37;
    ++*(_DWORD *)(a2 + 8);
    if ( v21 != &v39 )
      _libc_free(v21, v11);
    if ( *(_DWORD *)(a1 + 240) == 13 )
      goto LABEL_25;
  }
  v27 = a1 + 176;
  if ( !a4 )
  {
    v29 = *(_QWORD *)(a1 + 232);
    v36 = "unexpected ellipsis in argument list for ";
    v22 = 1;
    v40 = 771;
    v38 = "non-musttail call";
    sub_11FD800(v27, v29, (__int64)&v36, 1);
    return v22;
  }
  if ( !a5 )
  {
    v28 = *(_QWORD *)(a1 + 232);
    v36 = "unexpected ellipsis in argument list for ";
    v38 = "musttail call in non-varargs function";
    v40 = 771;
    sub_11FD800(v27, v28, (__int64)&v36, 1);
    return a4;
  }
  *(_DWORD *)(a1 + 240) = sub_1205200(v27);
  return sub_120AFE0(a1, 13, "expected ')' at end of argument list");
}
