// Function: sub_321D7C0
// Address: 0x321d7c0
//
__int64 __fastcall sub_321D7C0(__m128i *a1, __int64 *a2, __int64 a3)
{
  __int64 result; // rax
  __m128i *v4; // r12
  __int64 v5; // rax
  __int64 v6; // r14
  bool v7; // al
  __int64 v8; // r14
  __int64 v9; // rax
  bool v10; // al
  __int64 v11; // rax
  __int64 v12; // r12
  bool v13; // al
  __int64 v14; // rcx
  __int64 v15; // rdx
  __int64 v16; // rax
  bool v17; // al
  __int64 v18; // rax
  __int64 v19; // r14
  bool v20; // al
  __m128i v21; // xmm3
  __int64 v22; // rax
  __m128i *v23; // r12
  __m128i *v24; // r15
  bool v25; // dl
  __int64 v26; // r14
  bool v27; // al
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rdx
  __int64 v31; // rax
  __int64 v32; // r12
  __int64 v33; // r13
  __int64 v34; // rcx
  __int64 v35; // r8
  __m128i v36; // xmm6
  __int64 v37; // [rsp+8h] [rbp-98h]
  __m128i *v38; // [rsp+10h] [rbp-90h]
  __int64 v39; // [rsp+18h] [rbp-88h]
  __int64 *v40; // [rsp+20h] [rbp-80h]
  char v41[8]; // [rsp+30h] [rbp-70h] BYREF
  unsigned __int64 v42; // [rsp+38h] [rbp-68h]
  char v43; // [rsp+40h] [rbp-60h]
  char v44[8]; // [rsp+50h] [rbp-50h] BYREF
  unsigned __int64 v45; // [rsp+58h] [rbp-48h]
  bool v46; // [rsp+60h] [rbp-40h]

  result = (char *)a2 - (char *)a1;
  v38 = (__m128i *)a2;
  v37 = a3;
  if ( (char *)a2 - (char *)a1 <= 256 )
    return result;
  if ( !a3 )
  {
    v40 = a2;
    goto LABEL_59;
  }
  while ( 2 )
  {
    --v37;
    v4 = &a1[result >> 5];
    v5 = a1[1].m128i_i64[1];
    v6 = v4->m128i_i64[1];
    if ( v5 && v6 )
    {
      sub_AF47B0((__int64)v41, *(unsigned __int64 **)(v5 + 16), *(unsigned __int64 **)(v5 + 24));
      sub_AF47B0((__int64)v44, *(unsigned __int64 **)(v6 + 16), *(unsigned __int64 **)(v6 + 24));
      v7 = v46;
      if ( v43 )
      {
        if ( !v46 )
        {
          v8 = v38[-1].m128i_i64[1];
          goto LABEL_19;
        }
        v7 = v42 < v45;
      }
    }
    else
    {
      v7 = v6 != 0;
    }
    v8 = v38[-1].m128i_i64[1];
    if ( !v7 )
    {
LABEL_19:
      v16 = a1[1].m128i_i64[1];
      if ( v8 && v16 )
      {
        sub_AF47B0((__int64)v41, *(unsigned __int64 **)(v16 + 16), *(unsigned __int64 **)(v16 + 24));
        sub_AF47B0((__int64)v44, *(unsigned __int64 **)(v8 + 16), *(unsigned __int64 **)(v8 + 24));
        v17 = v46;
        if ( v43 )
        {
          if ( !v46 )
            goto LABEL_23;
          v17 = v42 < v45;
        }
      }
      else
      {
        v17 = v8 != 0;
      }
      if ( v17 )
      {
        v15 = a1->m128i_i64[1];
        v21 = _mm_loadu_si128(a1 + 1);
        a1[1].m128i_i64[0] = a1->m128i_i64[0];
        a1[1].m128i_i64[1] = v15;
        *a1 = v21;
        goto LABEL_32;
      }
LABEL_23:
      v18 = v4->m128i_i64[1];
      v19 = v38[-1].m128i_i64[1];
      if ( v18 && v19 )
      {
        sub_AF47B0((__int64)v41, *(unsigned __int64 **)(v18 + 16), *(unsigned __int64 **)(v18 + 24));
        sub_AF47B0((__int64)v44, *(unsigned __int64 **)(v19 + 16), *(unsigned __int64 **)(v19 + 24));
        v20 = v46;
        if ( v43 )
        {
          if ( !v46 )
          {
            v14 = a1->m128i_i64[0];
            v15 = a1->m128i_i64[1];
            goto LABEL_66;
          }
          v20 = v42 < v45;
        }
        v14 = a1->m128i_i64[0];
        v15 = a1->m128i_i64[1];
        if ( v20 )
          goto LABEL_17;
      }
      else
      {
        v14 = a1->m128i_i64[0];
        v15 = a1->m128i_i64[1];
        if ( v19 )
          goto LABEL_17;
      }
LABEL_66:
      *a1 = _mm_loadu_si128(v4);
      v4->m128i_i64[0] = v14;
      v4->m128i_i64[1] = v15;
      v15 = a1[1].m128i_i64[1];
      goto LABEL_32;
    }
    v9 = v4->m128i_i64[1];
    if ( v8 && v9 )
    {
      sub_AF47B0((__int64)v41, *(unsigned __int64 **)(v9 + 16), *(unsigned __int64 **)(v9 + 24));
      sub_AF47B0((__int64)v44, *(unsigned __int64 **)(v8 + 16), *(unsigned __int64 **)(v8 + 24));
      v10 = v46;
      if ( v43 )
      {
        if ( !v46 )
          goto LABEL_13;
        v10 = v42 < v45;
      }
    }
    else
    {
      v10 = v8 != 0;
    }
    if ( v10 )
    {
      v30 = a1->m128i_i64[0];
      v31 = a1->m128i_i64[1];
      *a1 = _mm_loadu_si128(v4);
      v4->m128i_i64[0] = v30;
      v4->m128i_i64[1] = v31;
      v15 = a1[1].m128i_i64[1];
      goto LABEL_32;
    }
LABEL_13:
    v11 = a1[1].m128i_i64[1];
    v12 = v38[-1].m128i_i64[1];
    if ( !v12 || !v11 )
    {
      v14 = a1->m128i_i64[0];
      v15 = a1->m128i_i64[1];
      if ( v12 )
        goto LABEL_17;
      goto LABEL_68;
    }
    sub_AF47B0((__int64)v41, *(unsigned __int64 **)(v11 + 16), *(unsigned __int64 **)(v11 + 24));
    sub_AF47B0((__int64)v44, *(unsigned __int64 **)(v12 + 16), *(unsigned __int64 **)(v12 + 24));
    v13 = v46;
    if ( v43 )
    {
      if ( !v46 )
      {
        v14 = a1->m128i_i64[0];
        v15 = a1->m128i_i64[1];
        goto LABEL_68;
      }
      v13 = v42 < v45;
    }
    v14 = a1->m128i_i64[0];
    v15 = a1->m128i_i64[1];
    if ( !v13 )
    {
LABEL_68:
      v36 = _mm_loadu_si128(a1 + 1);
      a1[1].m128i_i64[0] = v14;
      a1[1].m128i_i64[1] = v15;
      *a1 = v36;
      goto LABEL_32;
    }
LABEL_17:
    *a1 = _mm_loadu_si128(v38 - 1);
    v38[-1].m128i_i64[0] = v14;
    v38[-1].m128i_i64[1] = v15;
    v15 = a1[1].m128i_i64[1];
LABEL_32:
    v22 = a1->m128i_i64[1];
    v23 = a1 + 1;
    v24 = v38;
    while ( 1 )
    {
      v40 = (__int64 *)v23;
      if ( v15 && v22 )
      {
        v39 = v22;
        sub_AF47B0((__int64)v41, *(unsigned __int64 **)(v15 + 16), *(unsigned __int64 **)(v15 + 24));
        sub_AF47B0((__int64)v44, *(unsigned __int64 **)(v39 + 16), *(unsigned __int64 **)(v39 + 24));
        v25 = v46;
        v22 = a1->m128i_i64[1];
        if ( v43 )
        {
          if ( !v46 )
            goto LABEL_40;
          v25 = v42 < v45;
        }
      }
      else
      {
        v25 = v22 != 0;
      }
      if ( v25 )
        goto LABEL_33;
LABEL_40:
      for ( --v24; ; --v24 )
      {
        v26 = v24->m128i_i64[1];
        if ( v22 && v26 )
        {
          sub_AF47B0((__int64)v41, *(unsigned __int64 **)(v22 + 16), *(unsigned __int64 **)(v22 + 24));
          sub_AF47B0((__int64)v44, *(unsigned __int64 **)(v26 + 16), *(unsigned __int64 **)(v26 + 24));
          v27 = v46;
          if ( v43 )
          {
            if ( !v46 )
              break;
            v27 = v42 < v45;
          }
        }
        else
        {
          v27 = v26 != 0;
        }
        if ( !v27 )
          break;
        v22 = a1->m128i_i64[1];
      }
      if ( v23 >= v24 )
        break;
      v28 = v23->m128i_i64[1];
      v29 = v23->m128i_i64[0];
      *v23 = _mm_loadu_si128(v24);
      v24->m128i_i64[0] = v29;
      v24->m128i_i64[1] = v28;
      v22 = a1->m128i_i64[1];
LABEL_33:
      v15 = v23[1].m128i_i64[1];
      ++v23;
    }
    sub_321D7C0(v23, v38, v37);
    result = (char *)v23 - (char *)a1;
    if ( (char *)v23 - (char *)a1 > 256 )
    {
      if ( v37 )
      {
        v38 = v23;
        continue;
      }
LABEL_59:
      v32 = result >> 4;
      v33 = ((result >> 4) - 2) >> 1;
      sub_321B190((__int64)a1, v33, result >> 4, a1[v33].m128i_i64[0], a1[v33].m128i_i64[1]);
      do
      {
        --v33;
        sub_321B190((__int64)a1, v33, v32, a1[v33].m128i_i64[0], a1[v33].m128i_i64[1]);
      }
      while ( v33 );
      do
      {
        v40 -= 2;
        v34 = *v40;
        v35 = v40[1];
        *(__m128i *)v40 = _mm_loadu_si128(a1);
        result = sub_321B190((__int64)a1, 0, ((char *)v40 - (char *)a1) >> 4, v34, v35);
      }
      while ( (char *)v40 - (char *)a1 > 16 );
    }
    return result;
  }
}
