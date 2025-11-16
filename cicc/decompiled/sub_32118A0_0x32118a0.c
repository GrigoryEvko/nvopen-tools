// Function: sub_32118A0
// Address: 0x32118a0
//
__m128i *__fastcall sub_32118A0(__m128i *a1, __int64 a2)
{
  bool v3; // zf
  __int64 v4; // rdx
  __int64 v6; // rax
  __int64 v7; // r9
  unsigned __int64 *v8; // r13
  __int64 v9; // r8
  unsigned __int64 v10; // rax
  unsigned int v11; // eax
  unsigned int v12; // eax
  __int64 v13; // r14
  __int32 v14; // eax
  char *v15; // rdi
  unsigned __int64 *v16; // rax
  __m128i v17; // xmm0
  __int64 v18; // rax
  __int64 v19; // rax
  unsigned __int64 v20; // rcx
  __int64 v21; // rax
  unsigned __int64 v22; // rdx
  __int64 v23; // rcx
  unsigned int v24; // eax
  _BYTE *v25; // rax
  size_t v26; // rdx
  char *v27; // rsi
  __int32 v28; // eax
  unsigned __int64 v29; // [rsp+8h] [rbp-98h]
  __int64 v30; // [rsp+10h] [rbp-90h]
  __int64 v31; // [rsp+18h] [rbp-88h]
  __int64 v32; // [rsp+18h] [rbp-88h]
  __int64 v33; // [rsp+18h] [rbp-88h]
  __int64 v34; // [rsp+18h] [rbp-88h]
  unsigned __int64 *v35; // [rsp+28h] [rbp-78h] BYREF
  __int32 v36; // [rsp+30h] [rbp-70h]
  char *v37; // [rsp+38h] [rbp-68h] BYREF
  __int64 v38; // [rsp+40h] [rbp-60h]
  char src[8]; // [rsp+48h] [rbp-58h] BYREF
  __m128i v40; // [rsp+50h] [rbp-50h] BYREF
  __int64 v41; // [rsp+60h] [rbp-40h]

  v3 = *(_WORD *)(a2 + 68) == 14;
  v4 = *(_QWORD *)(a2 + 32);
  v37 = src;
  v38 = 0x100000000LL;
  LOBYTE(v41) = 0;
  if ( !v3 )
  {
    if ( -858993459 * (unsigned int)((40LL * (*(_DWORD *)(a2 + 40) & 0xFFFFFF) - 80) >> 3) == 1 && !*(_BYTE *)(v4 + 80) )
    {
      v4 += 80;
      goto LABEL_7;
    }
LABEL_3:
    a1[3].m128i_i8[8] = 0;
    return a1;
  }
  if ( *(_BYTE *)v4 )
    goto LABEL_3;
LABEL_7:
  v36 = *(_DWORD *)(v4 + 8);
  v6 = sub_2E891C0(a2);
  v3 = *(_WORD *)(a2 + 68) == 15;
  v8 = *(unsigned __int64 **)(v6 + 16);
  v9 = v6;
  v35 = v8;
  if ( !v3 )
    goto LABEL_8;
  if ( -858993459 * (unsigned int)((40LL * (*(_DWORD *)(a2 + 40) & 0xFFFFFF) - 80) >> 3) == 1 && *v8 == 4101 )
  {
    v33 = v6;
    v24 = sub_AF4160(&v35);
    v9 = v33;
    v8 += v24;
    v35 = v8;
LABEL_8:
    v30 = 0;
    if ( *(unsigned __int64 **)(v9 + 24) != v8 )
    {
      while ( 1 )
      {
        while ( 1 )
        {
          v10 = *v8;
          if ( *v8 == 35 )
          {
            v20 = v8[1];
            v8 = v35;
            v30 += v20;
            goto LABEL_17;
          }
          if ( *v8 <= 0x23 )
            break;
          if ( v10 != 4096 )
            goto LABEL_27;
          v19 = v8[1];
          v40.m128i_i64[0] = v8[2];
          v40.m128i_i64[1] = v19;
          if ( !(_BYTE)v41 )
            LOBYTE(v41) = 1;
          v8 = v35;
LABEL_17:
          v32 = v9;
          v12 = sub_AF4160(&v35);
          v9 = v32;
          v8 += v12;
          v35 = v8;
          if ( *(unsigned __int64 **)(v32 + 24) == v8 )
            goto LABEL_18;
        }
        if ( v10 == 6 )
        {
          v21 = (unsigned int)v38;
          v22 = (unsigned int)v38 + 1LL;
          if ( v22 > HIDWORD(v38) )
          {
            v34 = v9;
            sub_C8D5F0((__int64)&v37, src, v22, 8u, v9, v7);
            v21 = (unsigned int)v38;
            v9 = v34;
          }
          v23 = v30;
          v30 = 0;
          *(_QWORD *)&v37[8 * v21] = v23;
          v8 = v35;
          LODWORD(v38) = v38 + 1;
          goto LABEL_17;
        }
        if ( v10 != 16 )
          goto LABEL_27;
        v31 = v9;
        v29 = v8[1];
        v11 = sub_AF4160(&v35);
        v9 = v31;
        v8 += v11;
        v35 = v8;
        if ( *(unsigned __int64 **)(v31 + 24) == v8 )
          goto LABEL_17;
        if ( *v8 == 28 )
        {
          v30 -= (int)v29;
          goto LABEL_17;
        }
        if ( *v8 == 34 )
        {
          v30 += (int)v29;
          goto LABEL_17;
        }
      }
    }
LABEL_18:
    v13 = (unsigned int)v38;
    if ( *(_WORD *)(a2 + 68) == 14 )
    {
      v25 = *(_BYTE **)(a2 + 32);
      if ( v25[40] == 1 && !*v25 )
      {
        if ( (unsigned __int64)(unsigned int)v38 + 1 > HIDWORD(v38) )
        {
          sub_C8D5F0((__int64)&v37, src, (unsigned int)v38 + 1LL, 8u, v9, v7);
          v13 = (unsigned int)v38;
        }
        *(_QWORD *)&v37[8 * v13] = v30;
        LODWORD(v13) = v38 + 1;
        LODWORD(v38) = v38 + 1;
      }
    }
    v14 = v36;
    v15 = v37;
    a1[1].m128i_i64[0] = 0x100000000LL;
    a1->m128i_i32[0] = v14;
    v16 = &a1[1].m128i_u64[1];
    a1->m128i_i64[1] = (__int64)&a1[1].m128i_i64[1];
    if ( !(_DWORD)v13 )
      goto LABEL_20;
    if ( v15 != src )
    {
      v28 = HIDWORD(v38);
      a1->m128i_i64[1] = (__int64)v15;
      v15 = src;
      a1[1].m128i_i32[0] = v13;
      a1[1].m128i_i32[1] = v28;
      v37 = src;
      goto LABEL_20;
    }
    v26 = 8;
    v27 = src;
    if ( (_DWORD)v13 != 1 )
    {
      sub_C8D5F0((__int64)&a1->m128i_i64[1], &a1[1].m128i_u64[1], (unsigned int)v13, 8u, v9, (unsigned int)v13);
      v26 = 8LL * (unsigned int)v38;
      if ( !v26 )
        goto LABEL_46;
      v16 = (unsigned __int64 *)a1->m128i_i64[1];
      v27 = v37;
    }
    memcpy(v16, v27, v26);
LABEL_46:
    a1[1].m128i_i32[0] = v13;
    v15 = v37;
LABEL_20:
    v17 = _mm_loadu_si128(&v40);
    v18 = v41;
    a1[3].m128i_i8[8] = 1;
    a1[3].m128i_i64[0] = v18;
    a1[2] = v17;
    goto LABEL_28;
  }
LABEL_27:
  a1[3].m128i_i8[8] = 0;
  v15 = v37;
LABEL_28:
  if ( v15 != src )
    _libc_free((unsigned __int64)v15);
  return a1;
}
