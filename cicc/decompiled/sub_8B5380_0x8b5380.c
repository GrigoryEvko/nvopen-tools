// Function: sub_8B5380
// Address: 0x8b5380
//
_DWORD *__fastcall sub_8B5380(
        __m128i *a1,
        const __m128i *a2,
        __int64 *a3,
        __int64 *a4,
        __int64 a5,
        __int64 a6,
        int a7,
        int a8,
        char a9,
        int a10,
        _DWORD *a11,
        _DWORD *a12)
{
  bool v14; // r12
  __m128i *i; // rdi
  __m128i *v16; // rax
  int v17; // r13d
  int v18; // ebx
  __m128i *k; // r9
  __m128i *v20; // rdi
  int v21; // eax
  int v22; // r15d
  unsigned int v23; // r8d
  int v24; // eax
  _DWORD *result; // rax
  __m128i *v26; // rdi
  bool v27; // zf
  __m128i *v28; // rax
  int v29; // r12d
  bool v30; // r12
  __m128i *v31; // r12
  const __m128i *v32; // r12
  char v33; // [rsp+Bh] [rbp-65h]
  _BOOL4 v34; // [rsp+Ch] [rbp-64h]
  _BOOL4 v35; // [rsp+10h] [rbp-60h]
  int v36; // [rsp+14h] [rbp-5Ch]
  bool v38; // [rsp+18h] [rbp-58h]
  int v39; // [rsp+20h] [rbp-50h]
  char v40; // [rsp+24h] [rbp-4Ch]
  int v41; // [rsp+24h] [rbp-4Ch]
  int v42; // [rsp+24h] [rbp-4Ch]
  __m128i *v44; // [rsp+30h] [rbp-40h] BYREF
  __m128i *j; // [rsp+38h] [rbp-38h] BYREF

  v14 = 0;
  j = a1;
  v44 = (__m128i *)a2;
  if ( dword_4F077BC )
    v14 = qword_4F077A8 > 0x9CA3u;
  for ( i = j; i[8].m128i_i8[12] == 12; i = (__m128i *)i[10].m128i_i64[0] )
    ;
  v16 = v44;
  for ( j = i; v16[8].m128i_i8[12] == 12; v16 = (__m128i *)v16[10].m128i_i64[0] )
    ;
  v44 = v16;
  v17 = sub_8D32E0(i);
  if ( !v17 )
  {
    v18 = sub_8D32E0(v44);
    if ( !v18 )
    {
      if ( v14 )
      {
        v36 = 0;
        v14 = 0;
        v40 = 0;
LABEL_11:
        if ( unk_4D04268 )
        {
          k = j;
          v20 = v44;
          v33 = 0;
          v35 = 0;
          v34 = 0;
          v39 = v18;
          v18 = 0;
          goto LABEL_13;
        }
        v39 = v18;
        v18 = 0;
LABEL_48:
        for ( k = j; k[8].m128i_i8[12] == 12; k = (__m128i *)k[10].m128i_i64[0] )
          ;
        v20 = v44;
        for ( j = k; v20[8].m128i_i8[12] == 12; v20 = (__m128i *)v20[10].m128i_i64[0] )
          ;
        v44 = v20;
        v33 = 0;
        v35 = 0;
        v34 = 0;
LABEL_13:
        if ( !a10 )
          goto LABEL_14;
LABEL_36:
        if ( a8 )
        {
          v24 = sub_8B3500(v20, (__int64)k, a4, a6, 0x880u);
          a8 = 0;
        }
        else
        {
          v24 = sub_8B3500(v20, (__int64)k, a4, a6, 0x80u);
        }
        goto LABEL_38;
      }
      goto LABEL_55;
    }
    v36 = sub_8D3070(v44);
    v39 = 0;
    v44 = (__m128i *)sub_8D46C0(v44);
    v26 = v44;
    v40 = 1;
    if ( !v14 )
      goto LABEL_26;
    v30 = 0;
    goto LABEL_70;
  }
  v18 = sub_8D3070(j);
  v28 = (__m128i *)sub_8D46C0(j);
  j = v28;
  if ( !v14 )
  {
    v29 = sub_8D32E0(v44);
    if ( !v29 )
    {
LABEL_55:
      v39 = v18;
      v14 = v17 != 0;
      v36 = 0;
      v18 = 0;
      v40 = 0;
      goto LABEL_26;
    }
    v36 = sub_8D3070(v44);
    v39 = v18;
    v18 = v29;
    v44 = (__m128i *)sub_8D46C0(v44);
    goto LABEL_68;
  }
  v41 = sub_8D2310(v28);
  v30 = v41 != 0;
  v36 = sub_8D32E0(v44);
  if ( v36 )
  {
    v42 = sub_8D3070(v44);
    v39 = v18;
    v18 = v36;
    v44 = (__m128i *)sub_8D46C0(v44);
    v26 = v44;
    v36 = v42;
LABEL_70:
    if ( (unsigned int)sub_8D2310(v26) )
      v30 = 1;
    v40 = v30 && v17 != v18;
    if ( v40 )
    {
      if ( !v17 )
      {
        if ( (unsigned int)sub_8D2E30(j) )
        {
          v31 = (__m128i *)sub_8D46C0(j);
          if ( (unsigned int)sub_8D2310(v31) )
            j = v31;
        }
        v14 = 0;
LABEL_26:
        if ( !(v18 | v17) )
        {
          v18 = v39;
          goto LABEL_11;
        }
        goto LABEL_27;
      }
      goto LABEL_83;
    }
    v14 = v17 != 0;
    v40 = v18 != 0;
    if ( !v17 )
      goto LABEL_26;
LABEL_68:
    v33 = 1;
    v14 = 1;
    v40 = 1;
    goto LABEL_29;
  }
  if ( !v41 )
  {
    v39 = v18;
    v14 = 1;
    v18 = 0;
    v40 = 0;
    goto LABEL_26;
  }
  v39 = v18;
  v18 = 0;
LABEL_83:
  if ( (unsigned int)sub_8D2E30(v44) )
  {
    v32 = (const __m128i *)sub_8D46C0(v44);
    if ( (unsigned int)sub_8D2310(v32) )
      v44 = (__m128i *)v32;
  }
  v14 = v17 != 0;
  v40 = v18 != 0;
  if ( v18 )
    goto LABEL_68;
LABEL_27:
  if ( !unk_4D04268 )
    goto LABEL_48;
  v33 = 0;
LABEL_29:
  sub_73E8D0(&j, (const __m128i **)&v44);
  k = j;
  if ( j[8].m128i_i8[12] == 12 )
  {
    do
      k = (__m128i *)k[10].m128i_i64[0];
    while ( k[8].m128i_i8[12] == 12 );
    v34 = j != k;
  }
  else
  {
    v34 = 0;
  }
  v27 = v44[8].m128i_i8[12] == 12;
  j = k;
  v20 = v44;
  if ( v27 )
  {
    do
      v20 = (__m128i *)v20[10].m128i_i64[0];
    while ( v20[8].m128i_i8[12] == 12 );
    v35 = v44 != v20;
  }
  else
  {
    v35 = 0;
  }
  v44 = v20;
  if ( a10 )
    goto LABEL_36;
LABEL_14:
  v21 = sub_8B3500(k, (__int64)v20, a3, a5, (unsigned __int8)(a7 != 0) << 11);
  v22 = v21;
  if ( a8 )
  {
    v23 = 2048;
    v38 = v21 == 0;
  }
  else
  {
    v38 = v21 == 0;
    v23 = 0;
  }
  v24 = sub_8B3500(v44, (__int64)j, a4, a6, v23);
  if ( !v24 || v38 )
  {
    a8 = v22;
LABEL_38:
    *a11 &= a8;
    *a12 &= v24;
    goto LABEL_39;
  }
  if ( (!dword_4F077BC || (_DWORD)qword_4F077B4 || qword_4F077A8 > 0x9FC3u) && v36 != v39 && v33 )
  {
    if ( v39 )
      goto LABEL_23;
    goto LABEL_63;
  }
  if ( !v35 && v34 )
  {
LABEL_23:
    result = a12;
    *a12 = 0;
    return result;
  }
  if ( !v34 && v35 || ((a8 ^ 1) & a7) != 0 )
    goto LABEL_63;
  if ( ((a7 ^ 1) & a8) != 0 )
    goto LABEL_23;
LABEL_39:
  result = a11;
  if ( *a11 )
  {
    result = a12;
    if ( *a12 )
    {
      if ( (a9 & 1) != 0 )
      {
        if ( !v18 && v14 )
          goto LABEL_64;
        if ( v17 || !v40 )
          return result;
LABEL_63:
        result = a11;
LABEL_64:
        *result = 0;
      }
    }
  }
  return result;
}
