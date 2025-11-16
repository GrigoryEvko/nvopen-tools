// Function: sub_1569DC0
// Address: 0x1569dc0
//
__int64 __fastcall sub_1569DC0(__int64 a1, __int64 a2, unsigned __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v5; // r12
  const __m128i *v6; // rbx
  __int64 v7; // rax
  void *v8; // rdi
  size_t v9; // rdx
  size_t v10; // rdx
  _BYTE *v11; // rax
  _QWORD *v12; // r12
  size_t v13; // r15
  unsigned __int64 v14; // rdx
  unsigned __int64 v15; // rax
  __int64 v16; // rdx
  _BYTE *v17; // rsi
  __int64 v18; // [rsp+0h] [rbp-170h]
  size_t v19; // [rsp+18h] [rbp-158h]
  __int64 v20; // [rsp+20h] [rbp-150h]
  const __m128i *v21; // [rsp+28h] [rbp-148h]
  __int64 i; // [rsp+30h] [rbp-140h]
  _QWORD v23[2]; // [rsp+40h] [rbp-130h] BYREF
  __m128i v24; // [rsp+50h] [rbp-120h] BYREF
  void *src; // [rsp+60h] [rbp-110h] BYREF
  size_t n; // [rsp+68h] [rbp-108h]
  _QWORD v27[2]; // [rsp+70h] [rbp-100h] BYREF
  unsigned __int64 v28[2]; // [rsp+80h] [rbp-F0h] BYREF
  _BYTE v29[32]; // [rsp+90h] [rbp-E0h] BYREF
  _QWORD v30[2]; // [rsp+B0h] [rbp-C0h] BYREF
  unsigned __int64 v31; // [rsp+C0h] [rbp-B0h]
  _BYTE *v32; // [rsp+C8h] [rbp-A8h]
  int v33; // [rsp+D0h] [rbp-A0h]
  unsigned __int64 *v34; // [rsp+D8h] [rbp-98h]
  const __m128i *v35; // [rsp+E0h] [rbp-90h] BYREF
  __int64 v36; // [rsp+E8h] [rbp-88h]
  _BYTE v37[128]; // [rsp+F0h] [rbp-80h] BYREF

  result = a1 + 8;
  v5 = *(_QWORD *)(a1 + 16);
  for ( i = a1 + 8; i != v5; v5 = *(_QWORD *)(v5 + 8) )
  {
    if ( !v5 )
      BUG();
    if ( (*(_BYTE *)(v5 - 22) & 0x20) == 0 )
      continue;
    v20 = v5 - 56;
    result = sub_15E61A0(v5 - 56, a2, a3, a4);
    if ( a3 <= 0x15 )
      continue;
    a4 = *(_QWORD *)result ^ 0x202C415441445F5FLL;
    a2 = a4 | *(_QWORD *)(result + 8) ^ 0x635F636A626F5F5FLL;
    if ( a2 || *(_DWORD *)(result + 16) != 1768715361 || *(_WORD *)(result + 20) != 29811 )
      continue;
    v23[0] = result;
    v23[1] = a3;
    v35 = (const __m128i *)v37;
    v36 = 0x500000000LL;
    sub_16D2880(v23, &v35, 44, 0xFFFFFFFFLL, 1);
    v34 = v28;
    v28[0] = (unsigned __int64)v29;
    v30[0] = &unk_49EFC48;
    v28[1] = 0x2000000000LL;
    v33 = 1;
    v32 = 0;
    v31 = 0;
    v30[1] = 0;
    sub_16E7A40(v30, 0, 0, 0);
    v6 = v35;
    v21 = &v35[(unsigned int)v36];
    if ( v35 != v21 )
    {
      v18 = v5;
      do
      {
        while ( 1 )
        {
          v11 = v32;
          v24 = _mm_loadu_si128(v6);
          if ( (unsigned __int64)v32 >= v31 )
          {
            v12 = (_QWORD *)sub_16E7DE0(v30, 44);
          }
          else
          {
            v12 = v30;
            ++v32;
            *v11 = 44;
          }
          v13 = 0;
          v14 = sub_16D24E0(&v24, byte_3F15413, 6, 0);
          v15 = v24.m128i_u64[1];
          if ( v14 < v24.m128i_i64[1] )
          {
            v13 = v24.m128i_i64[1] - v14;
            v15 = v14;
          }
          src = (void *)(v24.m128i_i64[0] + v15);
          n = v13;
          v7 = sub_16D2680(&src, byte_3F15413, 6, -1);
          v8 = (void *)v12[3];
          v9 = v7 + 1;
          if ( v7 + 1 > n )
            v9 = n;
          v10 = n - v13 + v9;
          if ( v10 > n )
            v10 = n;
          if ( v12[2] - (_QWORD)v8 >= v10 )
            break;
          ++v6;
          sub_16E7EE0(v12, (const char *)src);
          if ( v21 == v6 )
            goto LABEL_26;
        }
        if ( v10 )
        {
          v19 = v10;
          memcpy(v8, src, v10);
          v12[3] += v19;
        }
        ++v6;
      }
      while ( v21 != v6 );
LABEL_26:
      v5 = v18;
    }
    v16 = *((unsigned int *)v34 + 2);
    v17 = (_BYTE *)*v34;
    if ( *((_DWORD *)v34 + 2) )
    {
      --v16;
      ++v17;
    }
    else if ( !v17 )
    {
      LOBYTE(v27[0]) = 0;
      src = v27;
      n = 0;
      goto LABEL_30;
    }
    src = v27;
    sub_1564140((__int64 *)&src, v17, (__int64)&v17[v16]);
LABEL_30:
    v30[0] = &unk_49EFD28;
    sub_16E7960(v30);
    if ( (_BYTE *)v28[0] != v29 )
      _libc_free(v28[0]);
    if ( v35 != (const __m128i *)v37 )
      _libc_free((unsigned __int64)v35);
    a2 = (__int64)src;
    result = sub_15E5D20(v20, src, n);
    if ( src != v27 )
    {
      a2 = v27[0] + 1LL;
      result = j_j___libc_free_0(src, v27[0] + 1LL);
    }
  }
  return result;
}
