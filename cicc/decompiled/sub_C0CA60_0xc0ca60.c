// Function: sub_C0CA60
// Address: 0xc0ca60
//
__int64 __fastcall sub_C0CA60(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // r12d
  unsigned int v4; // r14d
  __m128i v5; // xmm0
  int v6; // r15d
  unsigned int v7; // r9d
  __int64 v8; // r10
  __int64 v9; // r11
  unsigned int i; // r8d
  __int64 v11; // r13
  int v12; // ecx
  int v13; // eax
  int v14; // edx
  int v15; // esi
  __int64 v16; // rax
  const void *v17; // rsi
  bool v18; // al
  unsigned int v19; // r8d
  int v20; // eax
  __int64 v22; // [rsp+0h] [rbp-A0h]
  int v23; // [rsp+8h] [rbp-98h]
  unsigned int v24; // [rsp+Ch] [rbp-94h]
  __int64 v25; // [rsp+10h] [rbp-90h]
  unsigned int v26; // [rsp+18h] [rbp-88h]
  int v27; // [rsp+1Ch] [rbp-84h]
  __int64 v28; // [rsp+28h] [rbp-78h] BYREF
  __m128i v29; // [rsp+30h] [rbp-70h] BYREF
  size_t n[2]; // [rsp+50h] [rbp-50h] BYREF
  __int64 v31; // [rsp+60h] [rbp-40h]

  v3 = a3;
  v29.m128i_i64[1] = a3;
  v4 = *(_DWORD *)(a1 + 24);
  v29.m128i_i64[0] = a2;
  v29.m128i_i32[2] = a3;
  v5 = _mm_loadu_si128(&v29);
  v31 = 0;
  *(__m128i *)n = v5;
  if ( !v4 )
  {
    ++*(_QWORD *)a1;
    v28 = 0;
    goto LABEL_25;
  }
  v6 = HIDWORD(n[1]);
  v7 = v4 - 1;
  v8 = *(_QWORD *)(a1 + 8);
  v9 = 0;
  v27 = 1;
  for ( i = (v4 - 1) & HIDWORD(n[1]); ; i = v7 & v19 )
  {
    v11 = v8 + 24LL * i;
    v12 = *(_DWORD *)(v11 + 12);
    if ( v6 == v12 )
    {
      v17 = *(const void **)v11;
      v18 = n[0] == -1;
      if ( *(_QWORD *)v11 != -1 )
      {
        v18 = n[0] == -2;
        if ( v17 != (const void *)-2LL )
        {
          if ( LODWORD(n[1]) != *(_DWORD *)(v11 + 8) )
          {
            if ( !v6 )
              goto LABEL_23;
            goto LABEL_19;
          }
          v22 = v9;
          v23 = *(_DWORD *)(v11 + 12);
          v24 = i;
          v26 = v7;
          v25 = v8;
          if ( !LODWORD(n[1]) )
            return *(_QWORD *)(v11 + 16);
          v20 = memcmp((const void *)n[0], v17, LODWORD(n[1]));
          v9 = v22;
          v12 = v23;
          i = v24;
          v7 = v26;
          v8 = v25;
          v18 = v20 == 0;
        }
      }
      if ( v18 )
        return *(_QWORD *)(v11 + 16);
    }
    if ( !v12 )
      break;
LABEL_19:
    if ( v12 == 1 && *(_QWORD *)v11 == -2 && !v9 )
      v9 = v11;
LABEL_23:
    v19 = v27 + i;
    ++v27;
  }
  if ( *(_QWORD *)v11 != -1 )
    goto LABEL_23;
  v13 = *(_DWORD *)(a1 + 16);
  if ( v9 )
    v11 = v9;
  ++*(_QWORD *)a1;
  v14 = v13 + 1;
  v28 = v11;
  if ( 4 * (v13 + 1) < 3 * v4 )
  {
    if ( v4 - *(_DWORD *)(a1 + 20) - v14 <= v4 >> 3 )
    {
      v15 = v4;
      goto LABEL_11;
    }
    goto LABEL_12;
  }
LABEL_25:
  v15 = 2 * v4;
LABEL_11:
  sub_C0C780(a1, v15);
  sub_C0C4A0(a1, (char **)n, &v28);
  v11 = v28;
  v14 = *(_DWORD *)(a1 + 16) + 1;
LABEL_12:
  *(_DWORD *)(a1 + 16) = v14;
  if ( *(_DWORD *)(v11 + 12) || *(_QWORD *)v11 != -1 )
    --*(_DWORD *)(a1 + 20);
  *(__m128i *)v11 = _mm_loadu_si128((const __m128i *)n);
  *(_QWORD *)(v11 + 16) = v31;
  v16 = ((1LL << *(_BYTE *)(a1 + 44)) + *(_QWORD *)(a1 + 32) - 1) & -(1LL << *(_BYTE *)(a1 + 44));
  *(_QWORD *)(v11 + 16) = v16;
  *(_QWORD *)(a1 + 32) = (*(_DWORD *)(a1 + 40) != 6) + (unsigned __int64)v3 + v16;
  return *(_QWORD *)(v11 + 16);
}
