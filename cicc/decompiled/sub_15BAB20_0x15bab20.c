// Function: sub_15BAB20
// Address: 0x15bab20
//
_QWORD *__fastcall sub_15BAB20(__int64 a1, int a2)
{
  __int64 v3; // rbx
  __int64 *v4; // r15
  unsigned __int64 v5; // rdi
  _QWORD *result; // rax
  __int64 v7; // rdx
  __int64 *v8; // r14
  _QWORD *i; // rdx
  __int64 *j; // rbx
  __int64 v11; // rax
  int v12; // r13d
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rax
  unsigned int v16; // edx
  __int64 *v17; // rax
  __int64 v18; // rax
  unsigned __int64 v19; // rax
  int v20; // r13d
  unsigned int v21; // eax
  _QWORD *v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // rsi
  int v25; // r8d
  _QWORD *v26; // rdi
  int v27; // edi
  unsigned int v28; // r8d
  _QWORD *v29; // r9
  __int8 *v30; // rax
  __int64 v31; // rcx
  __m128i *v32; // r9
  char *v33; // r8
  __m128i v34; // xmm0
  __int64 v35; // rcx
  __m128i v36; // xmm1
  __m128i v37; // xmm2
  char *v38; // r11
  unsigned __int64 v39; // rax
  unsigned __int64 v40; // rsi
  unsigned __int64 v41; // rax
  unsigned __int64 v42; // rdx
  __m128i *v43; // rax
  _QWORD *k; // rdx
  __int64 v45; // [rsp+8h] [rbp-138h]
  __int64 v46; // [rsp+10h] [rbp-130h]
  __int64 v47; // [rsp+10h] [rbp-130h]
  signed __int64 v48; // [rsp+10h] [rbp-130h]
  char *dest; // [rsp+20h] [rbp-120h]
  char *desta; // [rsp+20h] [rbp-120h]
  __int64 *destb; // [rsp+20h] [rbp-120h]
  __int64 v52; // [rsp+28h] [rbp-118h]
  __m128i v53; // [rsp+30h] [rbp-110h] BYREF
  __m128i v54; // [rsp+40h] [rbp-100h] BYREF
  __m128i v55; // [rsp+50h] [rbp-F0h] BYREF
  __int64 v56; // [rsp+60h] [rbp-E0h]
  __int64 v57; // [rsp+70h] [rbp-D0h] BYREF
  __int64 src; // [rsp+78h] [rbp-C8h] BYREF
  __int64 v59; // [rsp+80h] [rbp-C0h]
  __int64 v60; // [rsp+88h] [rbp-B8h] BYREF
  __m128i v61[4]; // [rsp+90h] [rbp-B0h] BYREF
  __m128i v62; // [rsp+D0h] [rbp-70h] BYREF
  __m128i v63; // [rsp+E0h] [rbp-60h]
  __m128i v64; // [rsp+F0h] [rbp-50h]
  __int64 v65; // [rsp+100h] [rbp-40h]
  __int64 v66; // [rsp+108h] [rbp-38h]

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(__int64 **)(a1 + 8);
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
  result = (_QWORD *)sub_22077B0(8LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    v8 = &v4[v3];
    *(_QWORD *)(a1 + 16) = 0;
    for ( i = &result[v7]; i != result; ++result )
    {
      if ( result )
        *result = -8;
    }
    for ( j = v4; v8 != j; ++j )
    {
      v11 = *j;
      if ( *j != -16 && v11 != -8 )
      {
        v12 = *(_DWORD *)(a1 + 24);
        if ( !v12 )
        {
          MEMORY[0] = *j;
          BUG();
        }
        v52 = *(_QWORD *)(a1 + 8);
        v13 = *(_QWORD *)(v11 - 8LL * *(unsigned int *)(v11 + 8));
        v14 = *(_QWORD *)(v11 + 24);
        v59 = v13;
        v60 = v14;
        if ( *(_BYTE *)v13 == 1 )
        {
          v15 = *(_QWORD *)(v13 + 136);
          v16 = *(_DWORD *)(v15 + 32);
          v17 = *(__int64 **)(v15 + 24);
          if ( v16 > 0x40 )
            v18 = *v17;
          else
            v18 = (__int64)((_QWORD)v17 << (64 - (unsigned __int8)v16)) >> (64 - (unsigned __int8)v16);
          v61[0].m128i_i64[0] = v18;
          LODWORD(v19) = sub_15B15B0(v61, &v60);
        }
        else
        {
          if ( !byte_4F99930[0] && (unsigned int)sub_2207590(byte_4F99930) )
          {
            v42 = unk_4FA04C8;
            if ( !unk_4FA04C8 )
              v42 = 0xFF51AFD7ED558CCDLL;
            qword_4F99938 = v42;
            sub_2207640(byte_4F99930);
          }
          v66 = qword_4F99938;
          v57 = 0;
          v30 = sub_15B2320(v61, &v57, v61[0].m128i_i8, (unsigned __int64)&v62, v59);
          v31 = v57;
          v32 = v61;
          v33 = v30 + 8;
          src = v60;
          if ( v30 + 8 <= (__int8 *)&v62 )
          {
            *(_QWORD *)v30 = v60;
          }
          else
          {
            v46 = v57;
            dest = (char *)((char *)&v62 - v30);
            memcpy(v30, &src, (char *)&v62 - v30);
            if ( v46 )
            {
              sub_1593A20((unsigned __int64 *)&v62, v61);
              v38 = dest;
              v35 = v46 + 64;
            }
            else
            {
              sub_15938B0((unsigned __int64 *)&v53, v61[0].m128i_i64, v66);
              v34 = _mm_loadu_si128(&v53);
              v35 = 64;
              v36 = _mm_loadu_si128(&v54);
              v37 = _mm_loadu_si128(&v55);
              v65 = v56;
              v38 = dest;
              v62 = v34;
              v63 = v36;
              v64 = v37;
            }
            v47 = v35;
            desta = &v61[0].m128i_i8[8LL - (_QWORD)v38];
            if ( desta > (char *)&v62 )
              abort();
            v43 = (__m128i *)memcpy(v61, (char *)&src + (_QWORD)v38, 8LL - (_QWORD)v38);
            v33 = desta;
            v31 = v47;
            v32 = v43;
          }
          if ( v31 )
          {
            v45 = v31;
            v48 = v33 - (char *)v32;
            destb = (__int64 *)v32;
            sub_15AF6E0(v32->m128i_i8, v33, v62.m128i_i8);
            sub_1593A20((unsigned __int64 *)&v62, destb);
            v39 = v62.m128i_i64[0]
                - 0x622015F714C7D297LL
                * (((0x9DDFEA08EB382D69LL
                   * ((0x9DDFEA08EB382D69LL * (v65 ^ v64.m128i_i64[0]))
                    ^ v65
                    ^ ((0x9DDFEA08EB382D69LL * (v65 ^ v64.m128i_i64[0])) >> 47))) >> 47)
                 ^ (0x9DDFEA08EB382D69LL
                  * ((0x9DDFEA08EB382D69LL * (v65 ^ v64.m128i_i64[0]))
                   ^ v65
                   ^ ((0x9DDFEA08EB382D69LL * (v65 ^ v64.m128i_i64[0])) >> 47))))
                - 0x4B6D499041670D8DLL * (((unsigned __int64)(v48 + v45) >> 47) ^ (v48 + v45));
            v40 = (0x9DDFEA08EB382D69LL
                 * (((0x9DDFEA08EB382D69LL * (v64.m128i_i64[1] ^ v63.m128i_i64[1])) >> 47)
                  ^ (0x9DDFEA08EB382D69LL * (v64.m128i_i64[1] ^ v63.m128i_i64[1]))
                  ^ v64.m128i_i64[1]))
                ^ ((0x9DDFEA08EB382D69LL
                  * (((0x9DDFEA08EB382D69LL * (v64.m128i_i64[1] ^ v63.m128i_i64[1])) >> 47)
                   ^ (0x9DDFEA08EB382D69LL * (v64.m128i_i64[1] ^ v63.m128i_i64[1]))
                   ^ v64.m128i_i64[1])) >> 47);
            v41 = 0x9DDFEA08EB382D69LL
                * (((0x9DDFEA08EB382D69LL
                   * (v39
                    ^ (0xB492B66FBE98F273LL * (v62.m128i_i64[1] ^ ((unsigned __int64)v62.m128i_i64[1] >> 47))
                     + v63.m128i_i64[0]
                     - 0x622015F714C7D297LL * v40))) >> 47)
                 ^ (0x9DDFEA08EB382D69LL
                  * (v39
                   ^ (0xB492B66FBE98F273LL * (v62.m128i_i64[1] ^ ((unsigned __int64)v62.m128i_i64[1] >> 47))
                    + v63.m128i_i64[0]
                    - 0x622015F714C7D297LL * v40)))
                 ^ v39);
            v19 = 0x9DDFEA08EB382D69LL * ((v41 >> 47) ^ v41);
          }
          else
          {
            LODWORD(v19) = sub_1593600(v32, v33 - (char *)v32, v66);
          }
        }
        v20 = v12 - 1;
        v21 = v20 & v19;
        v22 = (_QWORD *)(v52 + 8LL * v21);
        v23 = *j;
        v24 = *v22;
        if ( *j != *v22 )
        {
          v25 = 1;
          v26 = 0;
          while ( v24 != -8 )
          {
            if ( v26 || v24 != -16 )
              v22 = v26;
            v27 = v25 + 1;
            v28 = v21 + v25;
            v21 = v20 & v28;
            v29 = (_QWORD *)(v52 + 8LL * (v20 & v28));
            v24 = *v29;
            if ( *v29 == v23 )
            {
              v22 = (_QWORD *)(v52 + 8LL * (v20 & v28));
              goto LABEL_25;
            }
            v25 = v27;
            v26 = v22;
            v22 = v29;
          }
          if ( v26 )
            v22 = v26;
        }
LABEL_25:
        *v22 = v23;
        ++*(_DWORD *)(a1 + 16);
      }
    }
    return (_QWORD *)j___libc_free_0(v4);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[*(unsigned int *)(a1 + 24)]; k != result; ++result )
    {
      if ( result )
        *result = -8;
    }
  }
  return result;
}
