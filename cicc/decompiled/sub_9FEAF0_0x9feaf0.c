// Function: sub_9FEAF0
// Address: 0x9feaf0
//
__int64 *__fastcall sub_9FEAF0(__int64 *a1, const __m128i *a2, __int64 a3, __int64 a4)
{
  unsigned __int64 v5; // rax
  __int64 v6; // rbx
  __int64 v7; // r15
  __int64 i; // rbx
  _BYTE *v9; // rdx
  unsigned __int64 v10; // rdx
  __int32 v12; // edx
  __int64 v13; // rax
  _QWORD *v14; // r8
  _QWORD *v15; // rdi
  _QWORD *v16; // rax
  _QWORD *v17; // r15
  _QWORD *v18; // rax
  __int64 v19; // rdi
  _QWORD *v20; // rbx
  unsigned int v21; // ecx
  unsigned int v22; // edx
  char v23; // cl
  _QWORD *v24; // rdx
  int v25; // ebx
  unsigned __int64 v26; // rdx
  unsigned __int64 v27; // rax
  _QWORD *v28; // rax
  __int64 v29; // rdx
  _QWORD *j; // rdx
  __int64 v31; // r14
  _BYTE *v32; // rdi
  _QWORD *v33; // rax
  __int64 v34; // [rsp+8h] [rbp-B8h] BYREF
  __int64 v35[2]; // [rsp+10h] [rbp-B0h] BYREF
  void (__fastcall *v36)(__int64 *, __int64 *, __int64); // [rsp+20h] [rbp-A0h]
  _QWORD v37[18]; // [rsp+30h] [rbp-90h] BYREF

  sub_9CE050((unsigned __int64 *)v35, (__int64)a2, a3, a4);
  v5 = v35[0] & 0xFFFFFFFFFFFFFFFELL;
  if ( (v35[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
    goto LABEL_16;
  v6 = a2[27].m128i_i64[1];
  a2[114].m128i_i8[9] = 1;
  v7 = *(_QWORD *)(v6 + 32);
  for ( i = v6 + 24; i != v7; v7 = *(_QWORD *)(v7 + 8) )
  {
    v9 = (_BYTE *)(v7 - 56);
    if ( !v7 )
      v9 = 0;
    sub_9FDC80(v35, (__int64)a2, v9);
    v5 = v35[0] & 0xFFFFFFFFFFFFFFFELL;
    if ( (v35[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
      goto LABEL_16;
  }
  v10 = a2[28].m128i_u64[1];
  if ( *(_OWORD *)&a2[28] != 0 )
  {
    if ( v10 < a2[28].m128i_i64[0] )
      v10 = a2[28].m128i_u64[0];
    memset(v37, 0, 0x58u);
    sub_9E7B10(&v34, a2, v10, 0, (__int64)v35);
    if ( LOBYTE(v37[10]) )
    {
      LOBYTE(v37[10]) = 0;
      if ( v37[8] )
        ((void (__fastcall *)(_QWORD *, _QWORD *, __int64))v37[8])(&v37[6], &v37[6], 3);
    }
    if ( LOBYTE(v37[5]) )
    {
      LOBYTE(v37[5]) = 0;
      if ( v37[3] )
        ((void (__fastcall *)(_QWORD *, _QWORD *, __int64))v37[3])(&v37[1], &v37[1], 3);
    }
    if ( LOBYTE(v37[0]) )
    {
      LOBYTE(v37[0]) = 0;
      if ( v36 )
        v36(v35, v35, 3);
    }
    v5 = v34 & 0xFFFFFFFFFFFFFFFELL;
    if ( (v34 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    {
LABEL_16:
      *a1 = v5 | 1;
      return a1;
    }
  }
  if ( !a2[107].m128i_i32[0] )
  {
    v12 = a2[101].m128i_i32[0];
    if ( !v12 )
    {
      ++a2[100].m128i_i64[0];
      goto LABEL_26;
    }
    v14 = (_QWORD *)a2[100].m128i_i64[1];
    v17 = &v14[2 * a2[101].m128i_u32[2]];
    if ( v14 == v17 )
      goto LABEL_38;
    v18 = (_QWORD *)a2[100].m128i_i64[1];
    while ( 1 )
    {
      v19 = *v18;
      v20 = v18;
      if ( *v18 != -8192 && v19 != -4096 )
        break;
      v18 += 2;
      if ( v17 == v18 )
        goto LABEL_38;
    }
    if ( v18 == v17 )
    {
LABEL_38:
      ++a2[100].m128i_i64[0];
    }
    else
    {
      while ( 1 )
      {
        v31 = *(_QWORD *)(v19 + 16);
        if ( v31 )
        {
          do
          {
            while ( 1 )
            {
              v32 = *(_BYTE **)(v31 + 24);
              if ( *v32 == 85 )
                break;
              v31 = *(_QWORD *)(v31 + 8);
              if ( !v31 )
                goto LABEL_57;
            }
            sub_A939D0(v32, v20[1]);
            v31 = *(_QWORD *)(v31 + 8);
          }
          while ( v31 );
LABEL_57:
          if ( *(_QWORD *)(*v20 + 16LL) )
            sub_BD84D0(*v20, v20[1]);
        }
        sub_B2E860();
        v20 += 2;
        if ( v20 == v17 )
          break;
        while ( *v20 == -4096 || *v20 == -8192 )
        {
          v20 += 2;
          if ( v17 == v20 )
            goto LABEL_63;
        }
        if ( v20 == v17 )
          break;
        v19 = *v20;
      }
LABEL_63:
      v12 = a2[101].m128i_i32[0];
      ++a2[100].m128i_i64[0];
      if ( !v12 )
      {
LABEL_26:
        if ( a2[101].m128i_i32[1] )
        {
          v13 = a2[101].m128i_u32[2];
          if ( (unsigned int)v13 <= 0x40 )
          {
            v14 = (_QWORD *)a2[100].m128i_i64[1];
LABEL_29:
            v15 = v14;
            v16 = &v14[2 * v13];
            if ( v14 != v16 )
            {
              do
              {
                *v15 = -4096;
                v15 += 2;
              }
              while ( v16 != v15 );
            }
            a2[101].m128i_i64[0] = 0;
            goto LABEL_32;
          }
          sub_C7D6A0(a2[100].m128i_i64[1], 16 * v13, 8);
          a2[100].m128i_i64[1] = 0;
          a2[101].m128i_i64[0] = 0;
          a2[101].m128i_i32[2] = 0;
        }
LABEL_32:
        sub_A84D20(a2[27].m128i_i64[1]);
        sub_A85B60(a2[27].m128i_i64[1]);
        sub_A84F90(a2[27].m128i_i64[1]);
        sub_A89620(a2[27].m128i_i64[1]);
        *a1 = 1;
        return a1;
      }
      v14 = (_QWORD *)a2[100].m128i_i64[1];
    }
    v21 = 4 * v12;
    v13 = a2[101].m128i_u32[2];
    if ( (unsigned int)(4 * v12) < 0x40 )
      v21 = 64;
    if ( (unsigned int)v13 <= v21 )
      goto LABEL_29;
    v22 = v12 - 1;
    if ( v22 )
    {
      _BitScanReverse(&v22, v22);
      v23 = 33 - (v22 ^ 0x1F);
      v24 = v14;
      v25 = 1 << v23;
      if ( 1 << v23 < 64 )
        v25 = 64;
      if ( (_DWORD)v13 == v25 )
      {
        a2[101].m128i_i64[0] = 0;
        v33 = &v14[2 * v13];
        do
        {
          if ( v24 )
            *v24 = -4096;
          v24 += 2;
        }
        while ( v33 != v24 );
        goto LABEL_32;
      }
    }
    else
    {
      v25 = 64;
    }
    sub_C7D6A0(v14, 16 * v13, 8);
    v26 = ((((((((4 * v25 / 3u + 1) | ((unsigned __int64)(4 * v25 / 3u + 1) >> 1)) >> 2)
             | (4 * v25 / 3u + 1)
             | ((unsigned __int64)(4 * v25 / 3u + 1) >> 1)) >> 4)
           | (((4 * v25 / 3u + 1) | ((unsigned __int64)(4 * v25 / 3u + 1) >> 1)) >> 2)
           | (4 * v25 / 3u + 1)
           | ((unsigned __int64)(4 * v25 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v25 / 3u + 1) | ((unsigned __int64)(4 * v25 / 3u + 1) >> 1)) >> 2)
           | (4 * v25 / 3u + 1)
           | ((unsigned __int64)(4 * v25 / 3u + 1) >> 1)) >> 4)
         | (((4 * v25 / 3u + 1) | ((unsigned __int64)(4 * v25 / 3u + 1) >> 1)) >> 2)
         | (4 * v25 / 3u + 1)
         | ((unsigned __int64)(4 * v25 / 3u + 1) >> 1)) >> 16;
    v27 = (v26
         | (((((((4 * v25 / 3u + 1) | ((unsigned __int64)(4 * v25 / 3u + 1) >> 1)) >> 2)
             | (4 * v25 / 3u + 1)
             | ((unsigned __int64)(4 * v25 / 3u + 1) >> 1)) >> 4)
           | (((4 * v25 / 3u + 1) | ((unsigned __int64)(4 * v25 / 3u + 1) >> 1)) >> 2)
           | (4 * v25 / 3u + 1)
           | ((unsigned __int64)(4 * v25 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v25 / 3u + 1) | ((unsigned __int64)(4 * v25 / 3u + 1) >> 1)) >> 2)
           | (4 * v25 / 3u + 1)
           | ((unsigned __int64)(4 * v25 / 3u + 1) >> 1)) >> 4)
         | (((4 * v25 / 3u + 1) | ((unsigned __int64)(4 * v25 / 3u + 1) >> 1)) >> 2)
         | (4 * v25 / 3u + 1)
         | ((unsigned __int64)(4 * v25 / 3u + 1) >> 1))
        + 1;
    a2[101].m128i_i32[2] = v27;
    v28 = (_QWORD *)sub_C7D670(16 * v27, 8);
    v29 = a2[101].m128i_u32[2];
    a2[101].m128i_i64[0] = 0;
    a2[100].m128i_i64[1] = (__int64)v28;
    for ( j = &v28[2 * v29]; j != v28; v28 += 2 )
    {
      if ( v28 )
        *v28 = -4096;
    }
    goto LABEL_32;
  }
  v35[0] = (__int64)"Never resolved function from blockaddress";
  LOWORD(v37[0]) = 259;
  sub_9C81F0(a1, (__int64)&a2->m128i_i64[1], (__int64)v35);
  return a1;
}
