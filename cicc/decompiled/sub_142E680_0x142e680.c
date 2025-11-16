// Function: sub_142E680
// Address: 0x142e680
//
void __fastcall sub_142E680(size_t *a1, __int64 a2)
{
  int v2; // ebx
  size_t v3; // rdx
  __int64 v4; // rax
  size_t *v5; // r13
  _BYTE *v6; // rax
  size_t *v7; // r14
  __int64 v8; // rbx
  _BYTE *v9; // rsi
  size_t *v10; // rax
  size_t v11; // r10
  size_t v12; // r13
  signed __int64 v13; // r15
  __int64 v14; // rax
  size_t *v15; // r12
  _BYTE *v16; // rax
  _BYTE *v17; // r11
  unsigned __int64 v18; // r8
  __int64 v19; // rax
  bool v20; // bl
  bool v21; // bl
  void *v22; // r8
  __int64 v23; // rsi
  __int64 v24; // rsi
  __int64 v25; // r13
  unsigned int v26; // ebx
  unsigned int v27; // eax
  _QWORD *v28; // rdi
  unsigned __int64 v29; // rdx
  unsigned __int64 v30; // rax
  _QWORD *v31; // rax
  __int64 v32; // rdx
  _QWORD *i; // rdx
  size_t v34; // r12
  __int64 v35; // rax
  void *v36; // rdi
  _QWORD *v37; // rax
  size_t *v38; // [rsp+0h] [rbp-90h]
  size_t v39; // [rsp+8h] [rbp-88h]
  unsigned __int64 v40; // [rsp+10h] [rbp-80h]
  size_t *v41; // [rsp+18h] [rbp-78h]
  signed __int64 v42; // [rsp+20h] [rbp-70h]
  size_t n; // [rsp+28h] [rbp-68h]
  size_t na; // [rsp+28h] [rbp-68h]
  __m128i v45; // [rsp+30h] [rbp-60h] BYREF
  void *src; // [rsp+40h] [rbp-50h] BYREF
  _BYTE *v47; // [rsp+48h] [rbp-48h]
  __int64 v48; // [rsp+50h] [rbp-40h]

  v2 = *((_DWORD *)a1 + 4);
  ++*a1;
  v38 = a1;
  if ( !v2 )
  {
    v3 = *((unsigned int *)a1 + 5);
    if ( !(_DWORD)v3 )
      return;
    v4 = *((unsigned int *)a1 + 6);
    if ( (unsigned int)v4 > 0x40 )
    {
      sub_142DC00(a1, a2, v3);
      if ( *((_DWORD *)a1 + 6) )
      {
        j___libc_free_0(a1[1]);
        a1[1] = 0;
        a1[2] = 0;
        *((_DWORD *)a1 + 6) = 0;
        return;
      }
      goto LABEL_56;
    }
LABEL_4:
    v45.m128i_i64[0] = 0;
    v45.m128i_i64[1] = -1;
    v5 = (size_t *)a1[1];
    src = 0;
    v47 = 0;
    v48 = 0;
    v41 = &v5[5 * v4];
    if ( v5 != v41 )
    {
      n = 0;
      v6 = 0;
      v7 = v5;
      v8 = -1;
      v9 = 0;
      while ( 1 )
      {
        v12 = v6 - v9;
        v13 = v6 - v9;
        if ( v6 == v9 )
        {
          v15 = 0;
        }
        else
        {
          if ( v12 > 0x7FFFFFFFFFFFFFF8LL )
            goto LABEL_64;
          a1 = (size_t *)(v6 - v9);
          v14 = sub_22077B0(v12);
          v9 = src;
          v15 = (size_t *)v14;
          v6 = v47;
          v12 = v47 - (_BYTE *)src;
        }
        if ( v9 != v6 )
        {
          a1 = v15;
          memmove(v15, v9, v12);
        }
        v16 = (_BYTE *)v7[3];
        v17 = (_BYTE *)v7[2];
        v3 = *v7;
        v9 = (_BYTE *)v7[1];
        v18 = v16 - v17;
        v42 = v16 - v17;
        if ( v16 == v17 )
        {
          v11 = 0;
          a1 = 0;
        }
        else
        {
          v39 = *v7;
          if ( v18 > 0x7FFFFFFFFFFFFFF8LL )
            goto LABEL_64;
          v19 = sub_22077B0(v18);
          v17 = (_BYTE *)v7[2];
          a1 = (size_t *)v19;
          v16 = (_BYTE *)v7[3];
          v3 = v39;
          v18 = v16 - v17;
          v11 = v16 - v17;
        }
        v20 = v3 == n && v8 == (_QWORD)v9;
        LOBYTE(v3) = v18 == v12;
        v21 = v18 == v12 && v20;
        if ( v16 != v17 )
          break;
        if ( v21 && v18 )
          goto LABEL_53;
        if ( a1 )
          goto LABEL_7;
LABEL_8:
        if ( v15 )
        {
          a1 = v15;
          j_j___libc_free_0(v15, v13);
        }
        if ( !v21 )
        {
          v9 = (_BYTE *)v7[2];
          v3 = v7[3] - (_QWORD)v9;
          v34 = v3;
          if ( v3 )
          {
            if ( v3 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_64:
              sub_4261EA(a1, v9, v3);
            v35 = sub_22077B0(v3);
            v9 = (_BYTE *)v7[2];
            v36 = (void *)v35;
            v3 = v7[3] - (_QWORD)v9;
            if ( (_BYTE *)v7[3] != v9 )
            {
LABEL_45:
              v36 = memmove(v36, v9, v3);
              goto LABEL_46;
            }
            if ( v35 )
LABEL_46:
              j_j___libc_free_0(v36, v34);
          }
          else
          {
            v36 = 0;
            if ( (_BYTE *)v7[3] != v9 )
              goto LABEL_45;
          }
          a1 = v7 + 2;
          *(__m128i *)v7 = _mm_loadu_si128(&v45);
          sub_142B670((__int64)(v7 + 2), (char **)&src);
        }
        v9 = src;
        v7 += 5;
        if ( v41 == v7 )
        {
          v22 = src;
          v23 = v48;
          v38[2] = 0;
          v24 = v23 - (_QWORD)v22;
          if ( v22 )
            j_j___libc_free_0(v22, v24);
          return;
        }
        v8 = v45.m128i_i64[1];
        n = v45.m128i_i64[0];
        v6 = v47;
      }
      v40 = v18;
      na = v11;
      v10 = (size_t *)memmove(a1, v17, v11);
      v11 = na;
      a1 = v10;
      if ( v21 && v40 )
LABEL_53:
        v21 = memcmp(a1, v15, v11) == 0;
LABEL_7:
      j_j___libc_free_0(a1, v42);
      goto LABEL_8;
    }
LABEL_56:
    a1[2] = 0;
    return;
  }
  v3 = (unsigned int)(4 * v2);
  v4 = *((unsigned int *)a1 + 6);
  if ( (unsigned int)v3 < 0x40 )
    v3 = 64;
  if ( (unsigned int)v3 >= (unsigned int)v4 )
    goto LABEL_4;
  v25 = 64;
  sub_142DC00(a1, a2, v3);
  v26 = v2 - 1;
  if ( v26 )
  {
    _BitScanReverse(&v27, v26);
    v25 = (unsigned int)(1 << (33 - (v27 ^ 0x1F)));
    if ( (int)v25 < 64 )
      v25 = 64;
  }
  v28 = (_QWORD *)a1[1];
  if ( (_DWORD)v25 == *((_DWORD *)v38 + 6) )
  {
    v38[2] = 0;
    v37 = &v28[5 * v25];
    do
    {
      if ( v28 )
      {
        *v28 = 0;
        v28[1] = -1;
        v28[2] = 0;
        v28[3] = 0;
        v28[4] = 0;
      }
      v28 += 5;
    }
    while ( v37 != v28 );
  }
  else
  {
    j___libc_free_0(v28);
    v29 = ((((((((4 * (int)v25 / 3u + 1) | ((unsigned __int64)(4 * (int)v25 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v25 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v25 / 3u + 1) >> 1)) >> 4)
           | (((4 * (int)v25 / 3u + 1) | ((unsigned __int64)(4 * (int)v25 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v25 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v25 / 3u + 1) >> 1)) >> 8)
         | (((((4 * (int)v25 / 3u + 1) | ((unsigned __int64)(4 * (int)v25 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v25 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v25 / 3u + 1) >> 1)) >> 4)
         | (((4 * (int)v25 / 3u + 1) | ((unsigned __int64)(4 * (int)v25 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v25 / 3u + 1)
         | ((unsigned __int64)(4 * (int)v25 / 3u + 1) >> 1)) >> 16;
    v30 = (v29
         | (((((((4 * (int)v25 / 3u + 1) | ((unsigned __int64)(4 * (int)v25 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v25 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v25 / 3u + 1) >> 1)) >> 4)
           | (((4 * (int)v25 / 3u + 1) | ((unsigned __int64)(4 * (int)v25 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v25 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v25 / 3u + 1) >> 1)) >> 8)
         | (((((4 * (int)v25 / 3u + 1) | ((unsigned __int64)(4 * (int)v25 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v25 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v25 / 3u + 1) >> 1)) >> 4)
         | (((4 * (int)v25 / 3u + 1) | ((unsigned __int64)(4 * (int)v25 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v25 / 3u + 1)
         | ((unsigned __int64)(4 * (int)v25 / 3u + 1) >> 1))
        + 1;
    *((_DWORD *)v38 + 6) = v30;
    v31 = (_QWORD *)sub_22077B0(40 * v30);
    v32 = *((unsigned int *)v38 + 6);
    v38[2] = 0;
    v38[1] = (size_t)v31;
    for ( i = &v31[5 * v32]; i != v31; v31 += 5 )
    {
      if ( v31 )
      {
        *v31 = 0;
        v31[1] = -1;
        v31[2] = 0;
        v31[3] = 0;
        v31[4] = 0;
      }
    }
  }
}
