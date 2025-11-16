// Function: sub_D043E0
// Address: 0xd043e0
//
__int64 __fastcall sub_D043E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 *v4; // r15
  __int64 v6; // rbx
  __int64 v7; // r9
  __int64 result; // rax
  __int64 v9; // r15
  __int64 v10; // r8
  __int64 v11; // r13
  __int64 v12; // rbx
  __int64 v13; // r14
  __int64 v14; // r15
  __int64 v15; // r12
  unsigned int v16; // r13d
  unsigned __int64 *v17; // rsi
  bool v18; // al
  __int64 v19; // rdx
  unsigned __int64 v20; // rcx
  unsigned __int64 v21; // rsi
  __int64 v22; // r8
  __m128i *v23; // r13
  int v24; // eax
  __m128i *v25; // r12
  __m128i v26; // xmm3
  unsigned int v27; // eax
  bool v28; // cc
  __int64 v29; // rdi
  __int64 v30; // rcx
  __int64 v31; // rsi
  __int64 v32; // r12
  __int64 v33; // r14
  int v34; // eax
  __int64 v35; // r13
  __m128i v36; // xmm1
  char v37; // al
  __int64 v38; // rdi
  int v39; // eax
  __int64 v40; // rax
  __int64 v41; // rax
  unsigned __int64 v42; // rax
  __int64 v43; // rdi
  unsigned int v44; // edx
  __int8 *v45; // r13
  __int64 v46; // [rsp+8h] [rbp-A8h]
  __int64 v48; // [rsp+20h] [rbp-90h]
  __int64 v49; // [rsp+28h] [rbp-88h]
  unsigned __int8 *v50; // [rsp+28h] [rbp-88h]
  __int64 v52; // [rsp+38h] [rbp-78h]
  __int64 v53; // [rsp+38h] [rbp-78h]
  __m128i v54; // [rsp+40h] [rbp-70h] BYREF
  __int128 v55; // [rsp+50h] [rbp-60h] BYREF
  __int128 v56; // [rsp+60h] [rbp-50h]
  __int64 v57; // [rsp+70h] [rbp-40h]

  v4 = (__int64 *)(a3 + 8);
  v6 = a2;
  if ( (int)sub_C49970(a2 + 8, (unsigned __int64 *)(a3 + 8)) < 0 )
    *(_DWORD *)(a2 + 264) &= ~4u;
  sub_C46B40(a2 + 8, v4);
  v7 = *(_QWORD *)(a3 + 24);
  v46 = a2 + 24;
  result = v7 + 56LL * *(unsigned int *)(a3 + 32);
  v48 = result;
  if ( v7 == result )
    return result;
  v9 = *(_QWORD *)(a3 + 24);
  do
  {
    v10 = *(_QWORD *)(v6 + 24);
    v11 = v10 + 56LL * *(unsigned int *)(v6 + 32);
    if ( v11 == v10 )
    {
LABEL_27:
      v57 = 256;
      v54 = 0;
      v55 = 0;
      v56 = 0;
      v54 = _mm_loadu_si128((const __m128i *)v9);
      *(_QWORD *)&v55 = *(_QWORD *)(v9 + 16);
      LODWORD(v56) = *(_DWORD *)(v9 + 32);
      if ( (unsigned int)v56 > 0x40 )
        sub_C43780((__int64)&v55 + 8, (const void **)(v9 + 24));
      else
        *((_QWORD *)&v55 + 1) = *(_QWORD *)(v9 + 24);
      v19 = *(unsigned int *)(v6 + 32);
      v20 = *(unsigned int *)(v6 + 36);
      v21 = *(_QWORD *)(v6 + 24);
      *((_QWORD *)&v56 + 1) = *(_QWORD *)(v9 + 40);
      v22 = v19 + 1;
      v23 = &v54;
      LOBYTE(v57) = *(_BYTE *)(v9 + 48);
      v24 = v19;
      if ( v19 + 1 > v20 )
      {
        if ( v21 > (unsigned __int64)&v54 || (unsigned __int64)&v54 >= v21 + 56 * v19 )
        {
          sub_D00C80(v46, v19 + 1, v19, (__int64)&v54, v22, v7);
          v19 = *(unsigned int *)(v6 + 32);
          v21 = *(_QWORD *)(v6 + 24);
          v23 = &v54;
          v24 = *(_DWORD *)(v6 + 32);
        }
        else
        {
          v45 = &v54.m128i_i8[-v21];
          sub_D00C80(v46, v19 + 1, v19, (__int64)v54.m128i_i64 - v21, v22, v7);
          v21 = *(_QWORD *)(v6 + 24);
          v19 = *(unsigned int *)(v6 + 32);
          v23 = (__m128i *)&v45[v21];
          v24 = *(_DWORD *)(v6 + 32);
        }
      }
      v25 = (__m128i *)(v21 + 56 * v19);
      if ( v25 )
      {
        v26 = _mm_loadu_si128(v23);
        v25[1].m128i_i64[0] = v23[1].m128i_i64[0];
        *v25 = v26;
        v27 = v23[2].m128i_u32[0];
        v25[2].m128i_i32[0] = v27;
        if ( v27 > 0x40 )
          sub_C43780((__int64)&v25[1].m128i_i64[1], (const void **)&v23[1].m128i_i64[1]);
        else
          v25[1].m128i_i64[1] = v23[1].m128i_i64[1];
        v25[2].m128i_i64[1] = v23[2].m128i_i64[1];
        v25[3].m128i_i16[0] = v23[3].m128i_i16[0];
        v24 = *(_DWORD *)(v6 + 32);
      }
      result = (unsigned int)(v24 + 1);
      *(_DWORD *)(v6 + 264) &= ~4u;
      v28 = (unsigned int)v56 <= 0x40;
      *(_DWORD *)(v6 + 32) = result;
      if ( !v28 )
      {
        v29 = *((_QWORD *)&v55 + 1);
        if ( *((_QWORD *)&v55 + 1) )
          goto LABEL_48;
      }
      goto LABEL_21;
    }
    v52 = v6;
    v12 = v9;
    v13 = 0;
    v14 = v10;
    while ( !(unsigned __int8)sub_D04110(a1, *(_QWORD *)v14, *(_QWORD *)v12, a4) )
    {
      v50 = *(unsigned __int8 **)v12;
      if ( (unsigned __int8)sub_D033B0(*(unsigned __int8 **)v14) )
      {
        if ( (unsigned __int8)sub_D033B0(v50) )
          break;
        v14 += 56;
        ++v13;
        if ( v11 == v14 )
        {
LABEL_26:
          v9 = v12;
          v6 = v52;
          goto LABEL_27;
        }
      }
      else
      {
LABEL_7:
        v14 += 56;
        ++v13;
        if ( v11 == v14 )
          goto LABEL_26;
      }
    }
    if ( *(_QWORD *)(*(_QWORD *)v14 + 8LL) != *(_QWORD *)(*(_QWORD *)v12 + 8LL)
      || (*(_QWORD *)(v14 + 8) != *(_QWORD *)(v12 + 8) || *(_DWORD *)(v14 + 16) != *(_DWORD *)(v12 + 16))
      && (!*(_BYTE *)(v14 + 20) && !*(_BYTE *)(v12 + 20)
       || *(_DWORD *)(v14 + 8) + *(_DWORD *)(v14 + 12) != *(_DWORD *)(v12 + 8) + *(_DWORD *)(v12 + 12)
       || *(_DWORD *)(v14 + 16) != *(_DWORD *)(v12 + 16)) )
    {
      goto LABEL_7;
    }
    v15 = v14;
    v49 = v13;
    v9 = v12;
    v6 = v52;
    v16 = *(_DWORD *)(v15 + 32);
    if ( !*(_BYTE *)(v15 + 49) )
      goto LABEL_16;
    v54.m128i_i32[2] = *(_DWORD *)(v15 + 32);
    if ( v16 <= 0x40 )
    {
      v41 = *(_QWORD *)(v15 + 24);
      goto LABEL_51;
    }
    sub_C43780((__int64)&v54, (const void **)(v15 + 24));
    v16 = v54.m128i_u32[2];
    if ( v54.m128i_i32[2] <= 0x40u )
    {
      v41 = v54.m128i_i64[0];
LABEL_51:
      v42 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v16) & ~v41;
      if ( !v16 )
        v42 = 0;
      v54.m128i_i64[0] = v42;
    }
    else
    {
      sub_C43D10((__int64)&v54);
    }
    sub_C46250((__int64)&v54);
    v16 = v54.m128i_u32[2];
    v54.m128i_i32[2] = 0;
    if ( *(_DWORD *)(v15 + 32) > 0x40u && (v43 = *(_QWORD *)(v15 + 24)) != 0 )
    {
      v53 = v54.m128i_i64[0];
      j_j___libc_free_0_0(v43);
      v44 = v54.m128i_u32[2];
      *(_DWORD *)(v15 + 32) = v16;
      *(_QWORD *)(v15 + 24) = v53;
      if ( v44 > 0x40 && v54.m128i_i64[0] )
      {
        j_j___libc_free_0_0(v54.m128i_i64[0]);
        v16 = *(_DWORD *)(v15 + 32);
      }
    }
    else
    {
      *(_QWORD *)(v15 + 24) = v54.m128i_i64[0];
      *(_DWORD *)(v15 + 32) = v16;
    }
    *(_WORD *)(v15 + 48) = 0;
LABEL_16:
    v17 = (unsigned __int64 *)(v9 + 24);
    if ( v16 <= 0x40 )
    {
      if ( *(_QWORD *)(v15 + 24) == *(_QWORD *)(v9 + 24) )
        goto LABEL_38;
LABEL_18:
      if ( (int)sub_C49970(v15 + 24, v17) < 0 )
        *(_DWORD *)(v6 + 264) &= ~4u;
      result = sub_C46B40(v15 + 24, (__int64 *)v17);
      *(_BYTE *)(v15 + 48) = 0;
    }
    else
    {
      v18 = sub_C43C50(v15 + 24, (const void **)v17);
      v17 = (unsigned __int64 *)(v9 + 24);
      if ( !v18 )
        goto LABEL_18;
LABEL_38:
      v30 = *(_QWORD *)(v6 + 24);
      v31 = *(unsigned int *)(v6 + 32);
      v32 = v30 + 56 * v13;
      v33 = v32 + 56;
      v34 = *(_DWORD *)(v6 + 32);
      v35 = 0x6DB6DB6DB6DB6DB7LL * ((56 * v31 - (56 * v49 + 56)) >> 3);
      if ( 56 * v31 - (56 * v49 + 56) > 0 )
      {
        while ( 1 )
        {
          v36 = _mm_loadu_si128((const __m128i *)(v32 + 56));
          v28 = *(_DWORD *)(v32 + 32) <= 0x40u;
          *(_DWORD *)(v32 + 16) = *(_DWORD *)(v32 + 72);
          v37 = *(_BYTE *)(v32 + 76);
          *(__m128i *)v32 = v36;
          *(_BYTE *)(v32 + 20) = v37;
          if ( !v28 )
          {
            v38 = *(_QWORD *)(v32 + 24);
            if ( v38 )
              j_j___libc_free_0_0(v38);
          }
          *(_QWORD *)(v32 + 24) = *(_QWORD *)(v32 + 80);
          v39 = *(_DWORD *)(v32 + 88);
          *(_DWORD *)(v32 + 88) = 0;
          *(_DWORD *)(v32 + 32) = v39;
          *(_QWORD *)(v32 + 40) = *(_QWORD *)(v32 + 96);
          *(_BYTE *)(v32 + 48) = *(_BYTE *)(v32 + 104);
          *(_BYTE *)(v32 + 49) = *(_BYTE *)(v32 + 105);
          v32 = v33;
          if ( !--v35 )
            break;
          v33 += 56;
        }
        v34 = *(_DWORD *)(v6 + 32);
        v30 = *(_QWORD *)(v6 + 24);
      }
      v40 = (unsigned int)(v34 - 1);
      *(_DWORD *)(v6 + 32) = v40;
      result = v30 + 56 * v40;
      if ( *(_DWORD *)(result + 32) > 0x40u )
      {
        v29 = *(_QWORD *)(result + 24);
        if ( v29 )
LABEL_48:
          result = j_j___libc_free_0_0(v29);
      }
    }
LABEL_21:
    v9 += 56;
  }
  while ( v48 != v9 );
  return result;
}
