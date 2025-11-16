// Function: sub_EA1BE0
// Address: 0xea1be0
//
__int64 __fastcall sub_EA1BE0(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // rdi
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  unsigned __int8 **v11; // rbx
  unsigned __int8 **v12; // r12
  unsigned __int8 *v13; // rsi
  const void *v14; // rsi
  __int64 v15; // rbx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 result; // rax
  __int64 v19; // rdi
  const __m128i *v20; // r12
  __int64 v21; // rax
  unsigned __int64 v22; // rdx
  __int64 v23; // rdx
  size_t v24; // r12
  __int64 v25; // rdi
  void *v26; // r13
  size_t v27; // r12
  _BYTE *v28; // rdi
  unsigned __int64 v29; // rcx
  __int8 *v30; // r12
  __int64 v31; // [rsp+8h] [rbp-1E8h]
  __int64 v32; // [rsp+10h] [rbp-1E0h]
  unsigned __int8 **v33; // [rsp+30h] [rbp-1C0h] BYREF
  __int64 v34; // [rsp+38h] [rbp-1B8h]
  _BYTE v35[96]; // [rsp+40h] [rbp-1B0h] BYREF
  void *src; // [rsp+A0h] [rbp-150h] BYREF
  size_t n; // [rsp+A8h] [rbp-148h]
  __int64 v38; // [rsp+B0h] [rbp-140h]
  _BYTE v39[312]; // [rsp+B8h] [rbp-138h] BYREF

  v5 = a1[37];
  v38 = 256;
  v33 = (unsigned __int8 **)v35;
  v34 = 0x400000000LL;
  v6 = *(_QWORD *)(v5 + 16);
  src = v39;
  n = 0;
  (*(void (__fastcall **)(__int64, __int64, void **, unsigned __int8 ***, __int64))(*(_QWORD *)v6 + 24LL))(
    v6,
    a2,
    &src,
    &v33,
    a3);
  v11 = v33;
  v12 = &v33[3 * (unsigned int)v34];
  if ( v33 != v12 )
  {
    do
    {
      v13 = *v11;
      v11 += 3;
      sub_EA1B60((__int64)a1, v13, v7, v8, v9, v10);
    }
    while ( v12 != v11 );
  }
  v14 = 0;
  v15 = sub_E8BB10(a1, 0);
  result = (unsigned int)v34;
  if ( (_DWORD)v34 )
  {
    v19 = v15 + 96;
    v16 = 0;
    v17 = 24LL * (unsigned int)v34;
    v14 = (const void *)(v15 + 112);
    do
    {
      v20 = (const __m128i *)((char *)v33 + v16);
      *(_DWORD *)((char *)v33 + v16 + 8) += *(_QWORD *)(v15 + 48);
      v21 = *(unsigned int *)(v15 + 104);
      v22 = v21 + 1;
      if ( v21 + 1 > (unsigned __int64)*(unsigned int *)(v15 + 108) )
      {
        v29 = *(_QWORD *)(v15 + 96);
        if ( v29 > (unsigned __int64)v20 )
        {
          v31 = v17;
          v32 = v16;
LABEL_21:
          sub_C8D5F0(v19, v14, v22, 0x18u, v16, v17);
          v23 = *(_QWORD *)(v15 + 96);
          v21 = *(unsigned int *)(v15 + 104);
          v16 = v32;
          v17 = v31;
          goto LABEL_7;
        }
        v31 = v17;
        v32 = v16;
        if ( (unsigned __int64)v20 >= v29 + 24 * v21 )
          goto LABEL_21;
        v30 = &v20->m128i_i8[-v29];
        sub_C8D5F0(v19, v14, v22, 0x18u, v16, v17);
        v23 = *(_QWORD *)(v15 + 96);
        v21 = *(unsigned int *)(v15 + 104);
        v17 = v31;
        v16 = v32;
        v20 = (const __m128i *)&v30[v23];
      }
      else
      {
        v23 = *(_QWORD *)(v15 + 96);
      }
LABEL_7:
      v16 += 24;
      result = v23 + 24 * v21;
      *(__m128i *)result = _mm_loadu_si128(v20);
      *(_QWORD *)(result + 16) = v20[1].m128i_i64[0];
      ++*(_DWORD *)(v15 + 104);
    }
    while ( v17 != v16 );
  }
  v24 = n;
  v25 = *(_QWORD *)(v15 + 48);
  *(_QWORD *)(v15 + 32) = a3;
  *(_BYTE *)(v15 + 29) |= 1u;
  v26 = src;
  if ( v24 + v25 > *(_QWORD *)(v15 + 56) )
  {
    v14 = (const void *)(v15 + 64);
    result = sub_C8D290(v15 + 40, (const void *)(v15 + 64), v24 + v25, 1u, v16, v17);
    v25 = *(_QWORD *)(v15 + 48);
  }
  if ( v24 )
  {
    v14 = v26;
    result = (__int64)memcpy((void *)(*(_QWORD *)(v15 + 40) + v25), v26, v24);
    v25 = *(_QWORD *)(v15 + 48);
  }
  v27 = v25 + v24;
  v28 = src;
  *(_QWORD *)(v15 + 48) = v27;
  if ( v28 != v39 )
    result = _libc_free(v28, v14);
  if ( v33 != (unsigned __int8 **)v35 )
    return _libc_free(v33, v14);
  return result;
}
