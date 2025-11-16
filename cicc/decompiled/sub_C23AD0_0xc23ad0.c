// Function: sub_C23AD0
// Address: 0xc23ad0
//
__int64 __fastcall sub_C23AD0(_QWORD *a1)
{
  _QWORD *v1; // rbx
  _QWORD *v2; // r13
  _QWORD *v3; // r14
  _QWORD *v4; // r12
  unsigned __int64 v5; // rsi
  char *v6; // r9
  __int64 v7; // r10
  __int64 v8; // rax
  __int64 result; // rax
  char *v10; // r8
  __int64 v11; // rdi
  int v12; // r15d
  int v13; // eax
  __int64 v14; // r12
  int v15; // r8d
  int v16; // r9d
  unsigned __int64 v17; // rcx
  unsigned __int64 v18; // rdi
  __int64 v19; // r10
  unsigned __int64 v20; // rdx
  unsigned __int64 v21; // r8
  const __m128i *v22; // rax
  __m128i *v23; // rdx
  char *v24; // r15
  __int64 v25; // r14
  __int64 v26; // rax
  __int64 v27; // r13
  __int64 v28; // r12
  __int64 v29; // r14
  char *v30; // r13
  char *v31; // r15
  void *v32; // rdi
  unsigned int v33; // ebx
  size_t v34; // rdx
  char *v35; // r12
  __int64 v36; // rdi
  char *v37; // [rsp+0h] [rbp-130h]
  __int64 *v38; // [rsp+8h] [rbp-128h]
  __int64 v39; // [rsp+10h] [rbp-120h]
  __int64 v40; // [rsp+18h] [rbp-118h]
  char *v41; // [rsp+20h] [rbp-110h]
  char *v43; // [rsp+28h] [rbp-108h]
  unsigned __int64 v44; // [rsp+30h] [rbp-100h] BYREF
  char v45; // [rsp+40h] [rbp-F0h]
  unsigned int v46; // [rsp+50h] [rbp-E0h] BYREF
  char v47; // [rsp+60h] [rbp-D0h]
  char *v48; // [rsp+70h] [rbp-C0h] BYREF
  __int64 v49; // [rsp+78h] [rbp-B8h]
  char v50; // [rsp+80h] [rbp-B0h]
  __int64 v51; // [rsp+90h] [rbp-A0h] BYREF
  char v52; // [rsp+A0h] [rbp-90h]
  __int64 v53; // [rsp+B0h] [rbp-80h] BYREF
  char v54; // [rsp+C0h] [rbp-70h]
  char *v55; // [rsp+D0h] [rbp-60h] BYREF
  __int64 v56; // [rsp+D8h] [rbp-58h]
  _DWORD v57[20]; // [rsp+E0h] [rbp-50h] BYREF

  v1 = a1;
  sub_C21E40((__int64)&v44, a1);
  if ( (v45 & 1) != 0 )
  {
    result = (unsigned int)v44;
    if ( (_DWORD)v44 )
      return result;
  }
  v2 = (_QWORD *)a1[31];
  v3 = (_QWORD *)a1[32];
  v38 = a1 + 31;
  if ( v2 != v3 )
  {
    v4 = (_QWORD *)a1[31];
    do
    {
      if ( (_QWORD *)*v4 != v4 + 2 )
        _libc_free(*v4, a1);
      v4 += 5;
    }
    while ( v3 != v4 );
    a1[32] = v2;
  }
  v5 = v44;
  if ( v44 > 0x333333333333333LL )
    sub_4262D8((__int64)"vector::reserve");
  v6 = (char *)a1[31];
  v7 = a1[33] - (_QWORD)v6;
  if ( v44 > 0xCCCCCCCCCCCCCCCDLL * (v7 >> 3) )
  {
    v24 = (char *)a1[32];
    v25 = 40 * v44;
    v43 = (char *)(v24 - v6);
    if ( v44 )
    {
      v41 = (char *)a1[31];
      v26 = sub_22077B0(40 * v44);
      v6 = v41;
      v27 = v26;
      if ( v41 == v24 )
      {
LABEL_61:
        v35 = (char *)v1[32];
        v24 = (char *)v1[31];
        if ( v35 == v24 )
        {
          v7 = v1[33] - (_QWORD)v24;
        }
        else
        {
          do
          {
            if ( *(char **)v24 != v24 + 16 )
              _libc_free(*(_QWORD *)v24, v5);
            v24 += 40;
          }
          while ( v35 != v24 );
          v24 = (char *)v1[31];
          v7 = v1[33] - (_QWORD)v24;
        }
        goto LABEL_66;
      }
    }
    else
    {
      v27 = 0;
      if ( v6 == v24 )
      {
LABEL_66:
        if ( v24 )
          j_j___libc_free_0(v24, v7);
        v1[31] = v27;
        v5 = v44;
        v1[32] = &v43[v27];
        v1[33] = v25 + v27;
        goto LABEL_10;
      }
    }
    v40 = 40 * v5;
    v28 = v27;
    v29 = v27;
    v30 = v24;
    v31 = v6;
    do
    {
      if ( v28 )
      {
        *(_DWORD *)(v28 + 8) = 0;
        v32 = (void *)(v28 + 16);
        *(_QWORD *)v28 = v28 + 16;
        *(_DWORD *)(v28 + 12) = 1;
        v33 = *((_DWORD *)v31 + 2);
        if ( v31 != (char *)v28 )
        {
          if ( v33 )
          {
            v34 = 24;
            if ( v33 == 1
              || (v5 = v28 + 16,
                  sub_C8D5F0(v28, v28 + 16, v33, 24),
                  v32 = *(void **)v28,
                  (v34 = 24LL * *((unsigned int *)v31 + 2)) != 0) )
            {
              v5 = *(_QWORD *)v31;
              memcpy(v32, *(const void **)v31, v34);
            }
            *(_DWORD *)(v28 + 8) = v33;
          }
        }
      }
      v31 += 40;
      v28 += 40;
    }
    while ( v30 != v31 );
    v27 = v29;
    v1 = a1;
    v25 = v40;
    goto LABEL_61;
  }
LABEL_10:
  if ( *((_BYTE *)v1 + 178) )
  {
    v8 = v1[34];
    if ( v1[35] == v8 )
    {
      if ( !v5 )
      {
LABEL_19:
        v1[37] = v8;
        goto LABEL_20;
      }
    }
    else
    {
      v1[35] = v8;
      if ( !v5 )
      {
        v1[37] = v8;
LABEL_14:
        sub_C1AFD0();
        return 0;
      }
    }
    sub_C22AA0((__int64)(v1 + 34), v5);
    v8 = v1[34];
    v5 = v44;
    goto LABEL_19;
  }
LABEL_20:
  if ( !v5 )
    goto LABEL_14;
  v39 = 0;
  while ( 1 )
  {
    v10 = (char *)v57;
    v11 = v1[32];
    v56 = 0x100000000LL;
    v55 = (char *)v57;
    if ( v11 == v1[33] )
    {
      v5 = v11;
      sub_C237A0(v38, (char *)v11, &v55);
      v10 = v55;
    }
    else
    {
      if ( v11 )
      {
        *(_QWORD *)(v11 + 8) = 0x100000000LL;
        *(_QWORD *)v11 = v11 + 16;
        if ( (_DWORD)v56 )
        {
          v5 = (unsigned __int64)&v55;
          sub_C1E9B0(v11, &v55);
        }
        v11 = v1[32];
        v10 = v55;
      }
      v1[32] = v11 + 40;
    }
    if ( v10 != (char *)v57 )
      _libc_free(v10, v5);
    v5 = (unsigned __int64)v1;
    sub_C22200((__int64)&v46, v1);
    if ( (v47 & 1) != 0 )
    {
      result = v46;
      if ( v46 )
        return result;
      goto LABEL_32;
    }
    if ( v46 )
      break;
LABEL_32:
    if ( v44 <= ++v39 )
      goto LABEL_14;
  }
  v12 = 0;
  while ( 1 )
  {
    sub_C21FD0((__int64)&v48, v1, 0);
    if ( (v50 & 1) != 0 )
    {
      result = (unsigned int)v48;
      if ( (_DWORD)v48 )
        return result;
    }
    sub_C21E40((__int64)&v51, v1);
    if ( (v52 & 1) != 0 )
    {
      result = (unsigned int)v51;
      if ( (_DWORD)v51 )
        return result;
    }
    if ( (v51 & 0xFFFF0000) != 0 )
    {
      sub_2241E40();
      return 0;
    }
    sub_C21E40((__int64)&v53, v1);
    if ( (v54 & 1) != 0 )
    {
      result = (unsigned int)v53;
      if ( (_DWORD)v53 )
        return result;
    }
    v14 = v1[32];
    v15 = v53;
    v16 = v51;
    v17 = *(unsigned int *)(v14 - 32);
    v5 = *(_QWORD *)(v14 - 40);
    v18 = *(unsigned int *)(v14 - 28);
    v19 = v49;
    v13 = *(_DWORD *)(v14 - 32);
    v20 = v5 + 24 * v17;
    if ( v17 < v18 )
    {
      if ( v20 )
      {
        *(_QWORD *)v20 = v48;
        *(_QWORD *)(v20 + 8) = v19;
        *(_DWORD *)(v20 + 16) = v16;
        *(_DWORD *)(v20 + 20) = v15;
        v13 = *(_DWORD *)(v14 - 32);
      }
      *(_DWORD *)(v14 - 32) = v13 + 1;
    }
    else
    {
      v57[1] = v53;
      v21 = v17 + 1;
      v22 = (const __m128i *)&v55;
      v55 = v48;
      v56 = v49;
      v57[0] = v51;
      if ( v18 < v17 + 1 )
      {
        v36 = v14 - 40;
        if ( v5 > (unsigned __int64)&v55 || v20 <= (unsigned __int64)&v55 )
        {
          sub_C8D5F0(v36, v14 - 24, v21, 24);
          v5 = *(_QWORD *)(v14 - 40);
          v17 = *(unsigned int *)(v14 - 32);
          v22 = (const __m128i *)&v55;
        }
        else
        {
          v37 = (char *)&v55 - v5;
          sub_C8D5F0(v36, v14 - 24, v21, 24);
          v5 = *(_QWORD *)(v14 - 40);
          v17 = *(unsigned int *)(v14 - 32);
          v22 = (const __m128i *)&v37[v5];
        }
      }
      v23 = (__m128i *)(v5 + 24 * v17);
      *v23 = _mm_loadu_si128(v22);
      v23[1].m128i_i64[0] = v22[1].m128i_i64[0];
      ++*(_DWORD *)(v14 - 32);
    }
    if ( v46 <= ++v12 )
      goto LABEL_32;
  }
}
