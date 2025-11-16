// Function: sub_2E05F00
// Address: 0x2e05f00
//
__int64 __fastcall sub_2E05F00(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v3; // rbx
  __int64 v4; // r13
  unsigned int v5; // eax
  __int64 result; // rax
  __int64 v7; // rcx
  __int64 v8; // r14
  const __m128i *v9; // r15
  __int64 i; // rsi
  int v11; // edx
  const __m128i *v12; // rbx
  __int8 v13; // al
  __int64 v14; // rax
  int v15; // eax
  int v16; // r11d
  __m128i *v17; // r10
  int v18; // r8d
  unsigned int j; // r9d
  __m128i *v20; // rcx
  __int8 v21; // dl
  __m128i v22; // xmm0
  __int64 k; // rcx
  int v24; // edx
  char v25; // al
  char v26; // al
  char v27; // al
  unsigned int v28; // r9d
  const __m128i *v29; // rsi
  __int64 v30; // rcx
  _DWORD *v31; // rdi
  int v32; // [rsp+Ch] [rbp-124h]
  int v33; // [rsp+Ch] [rbp-124h]
  __m128i *v34; // [rsp+10h] [rbp-120h]
  __m128i *v35; // [rsp+10h] [rbp-120h]
  unsigned int v36; // [rsp+18h] [rbp-118h]
  unsigned int v37; // [rsp+18h] [rbp-118h]
  int v38; // [rsp+1Ch] [rbp-114h]
  int v39; // [rsp+1Ch] [rbp-114h]
  int v40; // [rsp+20h] [rbp-110h]
  __m128i *v41; // [rsp+20h] [rbp-110h]
  __m128i *v42; // [rsp+20h] [rbp-110h]
  __int64 v43; // [rsp+28h] [rbp-108h]
  _QWORD v44[6]; // [rsp+40h] [rbp-F0h] BYREF
  _QWORD v45[6]; // [rsp+70h] [rbp-C0h] BYREF
  _QWORD v46[6]; // [rsp+A0h] [rbp-90h] BYREF
  _QWORD v47[12]; // [rsp+D0h] [rbp-60h] BYREF

  v2 = (unsigned int)(a2 - 1);
  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(_QWORD *)(a1 + 8);
  v5 = (((((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
        | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
        | ((v2 | (v2 >> 1)) >> 2)
        | v2
        | (v2 >> 1)) >> 16)
      | ((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
      | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
      | ((v2 | (v2 >> 1)) >> 2)
      | v2
      | (v2 >> 1))
     + 1;
  if ( v5 < 0x40 )
    v5 = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = sub_C7D670(48LL * v5, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v8 = 48 * v3;
    v9 = (const __m128i *)(v4 + 48 * v3);
    for ( i = result + 48 * v7; i != result; result += 48 )
    {
      if ( result )
      {
        v11 = *(_DWORD *)result;
        *(_QWORD *)(result + 16) = 0;
        *(_DWORD *)result = v11 & 0xFFF00000 | 0x15;
      }
    }
    v44[0] = 21;
    v44[2] = 0;
    v45[0] = 22;
    v45[2] = 0;
    if ( v9 == (const __m128i *)v4 )
      return sub_C7D6A0(v4, v8, 8);
    v12 = (const __m128i *)v4;
    while ( 2 )
    {
      while ( 1 )
      {
        v13 = v12->m128i_i8[0];
        if ( (unsigned __int8)(v12->m128i_i8[0] - 21) <= 1u )
          break;
        if ( (unsigned __int8)sub_2EAB6C0(v12, v44) )
          goto LABEL_11;
        v13 = v12->m128i_i8[0];
        if ( (unsigned __int8)(v12->m128i_i8[0] - 21) <= 1u )
          goto LABEL_22;
        if ( (unsigned __int8)sub_2EAB6C0(v12, v45) )
          goto LABEL_11;
LABEL_16:
        if ( !*(_DWORD *)(a1 + 24) )
        {
          v29 = v12;
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
        v14 = *(_QWORD *)(a1 + 8);
        v40 = *(_DWORD *)(a1 + 24);
        v46[0] = 21;
        v43 = v14;
        v46[2] = 0;
        v47[0] = 22;
        v47[2] = 0;
        v15 = sub_2EAE040(v12);
        v16 = 1;
        v17 = 0;
        v18 = v40 - 1;
        for ( j = (v40 - 1) & v15; ; j = v18 & v28 )
        {
          v20 = (__m128i *)(v43 + 48LL * j);
          if ( (unsigned __int8)(v12->m128i_i8[0] - 21) > 1u )
          {
            v32 = v16;
            v34 = v17;
            v36 = j;
            v38 = v18;
            v41 = (__m128i *)(v43 + 48LL * j);
            v25 = sub_2EAB6C0(v12, v41);
            v20 = v41;
            v18 = v38;
            j = v36;
            v17 = v34;
            v16 = v32;
            if ( v25 )
              goto LABEL_20;
            v21 = v41->m128i_i8[0];
          }
          else
          {
            v21 = v20->m128i_i8[0];
            if ( v12->m128i_i8[0] == v20->m128i_i8[0] )
              goto LABEL_20;
          }
          if ( (unsigned __int8)(v21 - 21) <= 1u )
          {
            if ( v21 == LOBYTE(v46[0]) )
              break;
LABEL_42:
            if ( LOBYTE(v47[0]) != v21 )
              goto LABEL_41;
            goto LABEL_39;
          }
          v33 = v16;
          v35 = v17;
          v37 = j;
          v39 = v18;
          v42 = v20;
          v26 = sub_2EAB6C0(v20, v46);
          v20 = v42;
          v18 = v39;
          j = v37;
          v17 = v35;
          v16 = v33;
          if ( v26 )
            break;
          v21 = v42->m128i_i8[0];
          if ( (unsigned __int8)(v42->m128i_i8[0] - 21) <= 1u )
            goto LABEL_42;
          v27 = sub_2EAB6C0(v42, v47);
          v20 = v42;
          v18 = v39;
          j = v37;
          v17 = v35;
          v16 = v33;
          if ( !v27 )
            goto LABEL_41;
LABEL_39:
          if ( !v17 )
            v17 = v20;
LABEL_41:
          v28 = v16 + j;
          ++v16;
        }
        if ( v17 )
          v20 = v17;
LABEL_20:
        v22 = _mm_loadu_si128(v12);
        v12 += 3;
        *v20 = v22;
        v20[1] = _mm_loadu_si128(v12 - 2);
        v20[2].m128i_i64[0] = v12[-1].m128i_i64[0];
        v20[2].m128i_i32[2] = v12[-1].m128i_i32[2];
        ++*(_DWORD *)(a1 + 16);
        if ( v9 == v12 )
          return sub_C7D6A0(v4, v8, 8);
      }
      if ( v13 == LOBYTE(v44[0]) )
        goto LABEL_11;
LABEL_22:
      if ( LOBYTE(v45[0]) == v13 )
      {
LABEL_11:
        v12 += 3;
        if ( v9 == v12 )
          return sub_C7D6A0(v4, v8, 8);
        continue;
      }
      goto LABEL_16;
    }
  }
  *(_QWORD *)(a1 + 16) = 0;
  for ( k = result + 48LL * *(unsigned int *)(a1 + 24); k != result; result += 48 )
  {
    if ( result )
    {
      v24 = *(_DWORD *)result;
      *(_QWORD *)(result + 16) = 0;
      *(_DWORD *)result = v24 & 0xFFF00000 | 0x15;
    }
  }
  return result;
}
