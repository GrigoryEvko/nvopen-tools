// Function: sub_1DB1790
// Address: 0x1db1790
//
__int64 __fastcall sub_1DB1790(__int64 a1, int a2)
{
  __int64 v3; // rbx
  const __m128i *v4; // r12
  unsigned __int64 v5; // rax
  __int64 result; // rax
  __int64 v7; // rsi
  const __m128i *v8; // r15
  __int64 i; // rdi
  int v10; // esi
  const __m128i *v11; // rbx
  __int8 v12; // al
  __int64 v13; // rax
  int v14; // eax
  int v15; // edx
  __m128i *v16; // r11
  int v17; // r9d
  unsigned int j; // r10d
  __m128i *v19; // r8
  __int8 v20; // si
  __m128i v21; // xmm0
  __int64 v22; // rdx
  __int64 k; // rcx
  int v24; // edx
  char v25; // al
  char v26; // al
  char v27; // al
  unsigned int v28; // r10d
  const __m128i *v29; // rsi
  __int64 v30; // rcx
  _DWORD *v31; // rdi
  int v32; // [rsp+4h] [rbp-11Ch]
  int v33; // [rsp+4h] [rbp-11Ch]
  __m128i *v34; // [rsp+8h] [rbp-118h]
  __m128i *v35; // [rsp+8h] [rbp-118h]
  unsigned int v36; // [rsp+10h] [rbp-110h]
  unsigned int v37; // [rsp+10h] [rbp-110h]
  int v38; // [rsp+14h] [rbp-10Ch]
  int v39; // [rsp+14h] [rbp-10Ch]
  int v40; // [rsp+18h] [rbp-108h]
  __m128i *v41; // [rsp+18h] [rbp-108h]
  __m128i *v42; // [rsp+18h] [rbp-108h]
  __int64 v43; // [rsp+20h] [rbp-100h]
  _QWORD v44[6]; // [rsp+30h] [rbp-F0h] BYREF
  _QWORD v45[6]; // [rsp+60h] [rbp-C0h] BYREF
  _QWORD v46[6]; // [rsp+90h] [rbp-90h] BYREF
  _QWORD v47[12]; // [rsp+C0h] [rbp-60h] BYREF

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(const __m128i **)(a1 + 8);
  v5 = ((((((((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
            | (unsigned int)(a2 - 1)
            | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
          | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 8)
        | (((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
        | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 16)
      | (((((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
        | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 8)
      | (((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
      | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
      | (unsigned int)(a2 - 1)
      | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1))
     + 1;
  if ( (unsigned int)v5 < 0x40 )
    LODWORD(v5) = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = sub_22077B0(48LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v8 = &v4[3 * v3];
    for ( i = result + 48 * v7; i != result; result += 48 )
    {
      if ( result )
      {
        v10 = *(_DWORD *)result;
        *(_QWORD *)(result + 16) = 0;
        *(_DWORD *)result = v10 & 0xFFF00000 | 0x13;
      }
    }
    v44[0] = 19;
    v44[2] = 0;
    v45[0] = 20;
    v45[2] = 0;
    if ( v8 == v4 )
      return j___libc_free_0(v4);
    v11 = v4;
    while ( 2 )
    {
      while ( 1 )
      {
        v12 = v11->m128i_i8[0];
        if ( (unsigned __int8)(v11->m128i_i8[0] - 19) <= 1u )
          break;
        if ( (unsigned __int8)sub_1E31610(v11, v44) )
          goto LABEL_11;
        v12 = v11->m128i_i8[0];
        if ( (unsigned __int8)(v11->m128i_i8[0] - 19) <= 1u )
          goto LABEL_22;
        if ( (unsigned __int8)sub_1E31610(v11, v45) )
          goto LABEL_11;
LABEL_16:
        if ( !*(_DWORD *)(a1 + 24) )
        {
          v29 = v11;
          v30 = 10;
          v31 = 0;
          while ( v30 )
          {
            *v31 = v29->m128i_i32[0];
            v29 = (const __m128i *)((char *)v29 + 4);
            ++v31;
            --v30;
          }
          BUG();
        }
        v13 = *(_QWORD *)(a1 + 8);
        v40 = *(_DWORD *)(a1 + 24);
        v46[0] = 19;
        v43 = v13;
        v46[2] = 0;
        v47[0] = 20;
        v47[2] = 0;
        v14 = sub_1E36300(v11);
        v15 = 1;
        v16 = 0;
        v17 = v40 - 1;
        for ( j = (v40 - 1) & v14; ; j = v17 & v28 )
        {
          v19 = (__m128i *)(v43 + 48LL * j);
          if ( (unsigned __int8)(v11->m128i_i8[0] - 19) > 1u )
          {
            v32 = v15;
            v34 = v16;
            v36 = j;
            v38 = v17;
            v41 = (__m128i *)(v43 + 48LL * j);
            v25 = sub_1E31610(v11, v41);
            v19 = v41;
            v17 = v38;
            j = v36;
            v16 = v34;
            v15 = v32;
            if ( v25 )
              goto LABEL_20;
            v20 = v41->m128i_i8[0];
          }
          else
          {
            v20 = v19->m128i_i8[0];
            if ( v11->m128i_i8[0] == v19->m128i_i8[0] )
              goto LABEL_20;
          }
          if ( (unsigned __int8)(v20 - 19) <= 1u )
          {
            if ( LOBYTE(v46[0]) == v20 )
              break;
LABEL_42:
            if ( v20 != LOBYTE(v47[0]) )
              goto LABEL_41;
            goto LABEL_39;
          }
          v33 = v15;
          v35 = v16;
          v37 = j;
          v39 = v17;
          v42 = v19;
          v26 = sub_1E31610(v19, v46);
          v19 = v42;
          v17 = v39;
          j = v37;
          v16 = v35;
          v15 = v33;
          if ( v26 )
            break;
          v20 = v42->m128i_i8[0];
          if ( (unsigned __int8)(v42->m128i_i8[0] - 19) <= 1u )
            goto LABEL_42;
          v27 = sub_1E31610(v42, v47);
          v19 = v42;
          v17 = v39;
          j = v37;
          v16 = v35;
          v15 = v33;
          if ( !v27 )
            goto LABEL_41;
LABEL_39:
          if ( !v16 )
            v16 = v19;
LABEL_41:
          v28 = v15 + j;
          ++v15;
        }
        if ( v16 )
          v19 = v16;
LABEL_20:
        v21 = _mm_loadu_si128(v11);
        v11 += 3;
        *v19 = v21;
        v19[1] = _mm_loadu_si128(v11 - 2);
        v19[2].m128i_i64[0] = v11[-1].m128i_i64[0];
        v19[2].m128i_i32[2] = v11[-1].m128i_i32[2];
        ++*(_DWORD *)(a1 + 16);
        if ( v8 == v11 )
          return j___libc_free_0(v4);
      }
      if ( v12 == LOBYTE(v44[0]) )
        goto LABEL_11;
LABEL_22:
      if ( v12 == LOBYTE(v45[0]) )
      {
LABEL_11:
        v11 += 3;
        if ( v8 == v11 )
          return j___libc_free_0(v4);
        continue;
      }
      goto LABEL_16;
    }
  }
  v22 = *(unsigned int *)(a1 + 24);
  *(_QWORD *)(a1 + 16) = 0;
  for ( k = result + 48 * v22; k != result; result += 48 )
  {
    if ( result )
    {
      v24 = *(_DWORD *)result;
      *(_QWORD *)(result + 16) = 0;
      *(_DWORD *)result = v24 & 0xFFF00000 | 0x13;
    }
  }
  return result;
}
