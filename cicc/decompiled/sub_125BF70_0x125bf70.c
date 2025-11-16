// Function: sub_125BF70
// Address: 0x125bf70
//
__m128i **__fastcall sub_125BF70(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rdx
  __m128i **result; // rax
  __int64 v5; // r13
  __int64 v7; // rbx
  __int64 v8; // rax
  unsigned int v9; // r9d
  int v10; // ecx
  int v11; // eax
  __int64 v12; // r15
  char *v13; // rdi
  bool v14; // al
  size_t v15; // rdx
  __m128i *v16; // r8
  __int64 *v17; // rax
  __int64 v18; // rdx
  __int64 *v19; // r8
  __int64 v20; // rdx
  __int64 v21; // rax
  unsigned __int16 *v22; // rsi
  size_t v23; // r15
  unsigned __int64 v24; // rax
  unsigned int v25; // eax
  unsigned __int64 v26; // rax
  __m128i *v27; // [rsp+8h] [rbp-68h]
  __int64 v28; // [rsp+8h] [rbp-68h]
  __int64 *v29; // [rsp+8h] [rbp-68h]
  __int64 *v30; // [rsp+8h] [rbp-68h]
  unsigned int v31; // [rsp+8h] [rbp-68h]
  __int64 v32; // [rsp+10h] [rbp-60h] BYREF
  __int64 v33; // [rsp+18h] [rbp-58h] BYREF
  __m128i *v34; // [rsp+20h] [rbp-50h] BYREF
  void *s2; // [rsp+28h] [rbp-48h]
  __int64 v36; // [rsp+30h] [rbp-40h]

  v3 = a3 << 6;
  result = &v34;
  v5 = a2 + v3;
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  if ( a2 + v3 != a2 )
  {
    v7 = a2;
    while ( 1 )
    {
      while ( 1 )
      {
        result = (__m128i **)sub_C6BF30(a1, v7, &v32);
        if ( !(_BYTE)result )
          break;
        v7 += 64;
        if ( v5 == v7 )
          return result;
      }
      v8 = v32;
      v9 = *(_DWORD *)(a1 + 24);
      ++*(_QWORD *)a1;
      v33 = v8;
      v11 = *(_DWORD *)(a1 + 16) + 1;
      if ( 4 * v11 >= 3 * v9 )
        break;
      if ( v9 - *(_DWORD *)(a1 + 20) - v11 <= v9 >> 3 )
      {
        v23 = *(_QWORD *)(a1 + 8);
        v31 = v9;
        v26 = ((((((((((v9 - 1) | ((unsigned __int64)(v9 - 1) >> 1)) >> 2) | (v9 - 1)
                                                                           | ((unsigned __int64)(v9 - 1) >> 1)) >> 4)
                 | (((v9 - 1) | ((unsigned __int64)(v9 - 1) >> 1)) >> 2)
                 | (v9 - 1)
                 | ((unsigned __int64)(v9 - 1) >> 1)) >> 8)
               | (((((v9 - 1) | ((unsigned __int64)(v9 - 1) >> 1)) >> 2) | (v9 - 1) | ((unsigned __int64)(v9 - 1) >> 1)) >> 4)
               | (((v9 - 1) | ((unsigned __int64)(v9 - 1) >> 1)) >> 2)
               | (v9 - 1)
               | ((unsigned __int64)(v9 - 1) >> 1)) >> 16)
             | (((((((v9 - 1) | ((unsigned __int64)(v9 - 1) >> 1)) >> 2) | (v9 - 1) | ((unsigned __int64)(v9 - 1) >> 1)) >> 4)
               | (((v9 - 1) | ((unsigned __int64)(v9 - 1) >> 1)) >> 2)
               | (v9 - 1)
               | ((unsigned __int64)(v9 - 1) >> 1)) >> 8)
             | (((((v9 - 1) | ((unsigned __int64)(v9 - 1) >> 1)) >> 2) | (v9 - 1) | ((unsigned __int64)(v9 - 1) >> 1)) >> 4)
             | (((v9 - 1) | ((unsigned __int64)(v9 - 1) >> 1)) >> 2)
             | (v9 - 1)
             | ((unsigned __int64)(v9 - 1) >> 1))
            + 1;
        if ( (unsigned int)v26 < 0x40 )
          LODWORD(v26) = 64;
        *(_DWORD *)(a1 + 24) = v26;
        *(_QWORD *)(a1 + 8) = sub_C7D670((unsigned __int64)(unsigned int)v26 << 6, 8);
        if ( !v23 )
          goto LABEL_37;
LABEL_32:
        sub_C6D3C0(a1, v23, v23 + ((unsigned __int64)v31 << 6));
        sub_C7D6A0(v23, (unsigned __int64)v31 << 6, 8);
        goto LABEL_33;
      }
LABEL_7:
      *(_DWORD *)(a1 + 16) = v11;
      sub_124B680(&v34, (char *)0xFFFFFFFFFFFFFFFFLL, 0);
      v12 = v33;
      v13 = *(char **)(v33 + 8);
      v14 = v13 + 1 == 0;
      if ( s2 == (void *)-1LL )
        goto LABEL_12;
      v14 = v13 + 2 == 0;
      if ( s2 == (void *)-2LL )
        goto LABEL_12;
      v15 = *(_QWORD *)(v33 + 16);
      if ( v36 != v15 )
        goto LABEL_27;
      if ( v15 )
      {
        v14 = memcmp(v13, s2, v15) == 0;
LABEL_12:
        if ( !v14 )
LABEL_27:
          --*(_DWORD *)(a1 + 20);
      }
      v16 = v34;
      if ( v34 )
      {
        if ( (__m128i *)v34->m128i_i64[0] != &v34[1] )
        {
          v27 = v34;
          j_j___libc_free_0(v34->m128i_i64[0], v34[1].m128i_i64[0] + 1);
          v16 = v27;
        }
        j_j___libc_free_0(v16, 32);
      }
      v28 = *(_QWORD *)v7;
      if ( *(_QWORD *)v7 )
      {
        v17 = (__int64 *)sub_22077B0(32);
        if ( v17 )
        {
          v18 = v28;
          v29 = v17;
          *v17 = (__int64)(v17 + 2);
          sub_125BAC0(v17, *(_BYTE **)v18, *(_QWORD *)v18 + *(_QWORD *)(v18 + 8));
          v17 = v29;
        }
        v19 = *(__int64 **)v12;
        *(_QWORD *)v12 = v17;
        if ( v19 )
        {
          if ( (__int64 *)*v19 != v19 + 2 )
          {
            v30 = v19;
            j_j___libc_free_0(*v19, v19[2] + 1);
            v19 = v30;
          }
          j_j___libc_free_0(v19, 32);
          v17 = *(__int64 **)v12;
        }
        v20 = *v17;
        v21 = v17[1];
        *(_QWORD *)(v12 + 8) = v20;
        *(_QWORD *)(v12 + 16) = v21;
      }
      else
      {
        *(__m128i *)(v12 + 8) = _mm_loadu_si128((const __m128i *)(v7 + 8));
      }
      v22 = (unsigned __int16 *)(v7 + 24);
      v7 += 64;
      *(_WORD *)(v12 + 24) = 0;
      result = (__m128i **)sub_C6A4F0(v12 + 24, v22);
      if ( v5 == v7 )
        return result;
    }
    v23 = *(_QWORD *)(a1 + 8);
    v31 = v9;
    v10 = 2 * v9;
    v24 = (((((((((unsigned int)(v10 - 1) | ((unsigned __int64)(unsigned int)(v10 - 1) >> 1)) >> 2)
              | (unsigned int)(v10 - 1)
              | ((unsigned __int64)(unsigned int)(v10 - 1) >> 1)) >> 4)
            | (((unsigned int)(v10 - 1) | ((unsigned __int64)(unsigned int)(v10 - 1) >> 1)) >> 2)
            | (unsigned int)(v10 - 1)
            | ((unsigned __int64)(unsigned int)(v10 - 1) >> 1)) >> 8)
          | (((((unsigned int)(v10 - 1) | ((unsigned __int64)(unsigned int)(v10 - 1) >> 1)) >> 2)
            | (unsigned int)(v10 - 1)
            | ((unsigned __int64)(unsigned int)(v10 - 1) >> 1)) >> 4)
          | (((unsigned int)(v10 - 1) | ((unsigned __int64)(unsigned int)(v10 - 1) >> 1)) >> 2)
          | (unsigned int)(v10 - 1)
          | ((unsigned __int64)(unsigned int)(v10 - 1) >> 1)) >> 16)
        | (((((((unsigned int)(v10 - 1) | ((unsigned __int64)(unsigned int)(v10 - 1) >> 1)) >> 2)
            | (unsigned int)(v10 - 1)
            | ((unsigned __int64)(unsigned int)(v10 - 1) >> 1)) >> 4)
          | (((unsigned int)(v10 - 1) | ((unsigned __int64)(unsigned int)(v10 - 1) >> 1)) >> 2)
          | (unsigned int)(v10 - 1)
          | ((unsigned __int64)(unsigned int)(v10 - 1) >> 1)) >> 8)
        | (((((unsigned int)(v10 - 1) | ((unsigned __int64)(unsigned int)(v10 - 1) >> 1)) >> 2)
          | (unsigned int)(v10 - 1)
          | ((unsigned __int64)(unsigned int)(v10 - 1) >> 1)) >> 4)
        | (((unsigned int)(v10 - 1) | ((unsigned __int64)(unsigned int)(v10 - 1) >> 1)) >> 2)
        | (unsigned int)(v10 - 1)
        | ((unsigned __int64)(unsigned int)(v10 - 1) >> 1);
    v25 = v24 + 1;
    if ( v25 < 0x40 )
      v25 = 64;
    *(_DWORD *)(a1 + 24) = v25;
    *(_QWORD *)(a1 + 8) = sub_C7D670((unsigned __int64)v25 << 6, 8);
    if ( v23 )
      goto LABEL_32;
LABEL_37:
    sub_C6BD30(a1);
LABEL_33:
    sub_C6BF30(a1, v7, &v33);
    v11 = *(_DWORD *)(a1 + 16) + 1;
    goto LABEL_7;
  }
  return result;
}
