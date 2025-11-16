// Function: sub_3988120
// Address: 0x3988120
//
__int64 __fastcall sub_3988120(__m128i *a1, __int64 *a2, __int64 a3)
{
  __int64 result; // rax
  __m128i *v4; // r12
  __int64 v5; // rax
  __int64 v6; // r14
  bool v7; // al
  __int64 v8; // r14
  __int64 v9; // rax
  bool v10; // al
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  __m128i *v15; // r12
  __m128i *v16; // r15
  bool v17; // dl
  bool v18; // al
  __int64 v19; // r14
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rax
  bool v23; // al
  __m128i v24; // xmm3
  __int64 v25; // r12
  __int64 v26; // r13
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // rax
  __int64 v30; // r12
  __int64 v31; // rax
  __m128i v32; // xmm6
  __int64 v33; // rax
  __int64 v34; // r14
  __int64 v35; // rdx
  __int64 v36; // rax
  bool v37; // cl
  bool v38; // cl
  __int64 v39; // [rsp+8h] [rbp-98h]
  __m128i *v40; // [rsp+10h] [rbp-90h]
  __int64 v41; // [rsp+18h] [rbp-88h]
  __int64 *v42; // [rsp+20h] [rbp-80h]
  char v43[8]; // [rsp+30h] [rbp-70h] BYREF
  unsigned __int64 v44; // [rsp+38h] [rbp-68h]
  char v45; // [rsp+40h] [rbp-60h]
  char v46[8]; // [rsp+50h] [rbp-50h] BYREF
  unsigned __int64 v47; // [rsp+58h] [rbp-48h]
  bool v48; // [rsp+60h] [rbp-40h]

  result = (char *)a2 - (char *)a1;
  v40 = (__m128i *)a2;
  v39 = a3;
  if ( (char *)a2 - (char *)a1 <= 256 )
    return result;
  if ( !a3 )
  {
    v42 = a2;
    goto LABEL_51;
  }
  while ( 2 )
  {
    --v39;
    v4 = &a1[result >> 5];
    v5 = a1[1].m128i_i64[1];
    v6 = v4->m128i_i64[1];
    if ( v5 && v6 )
    {
      sub_15B1350((__int64)v43, *(unsigned __int64 **)(v5 + 24), *(unsigned __int64 **)(v5 + 32));
      sub_15B1350((__int64)v46, *(unsigned __int64 **)(v6 + 24), *(unsigned __int64 **)(v6 + 32));
      if ( v45 )
      {
        if ( !v48 )
        {
          v8 = v40[-1].m128i_i64[1];
          goto LABEL_43;
        }
        v7 = v44 < v47;
      }
      else
      {
        v7 = v48;
      }
    }
    else
    {
      v7 = v6 != 0;
    }
    v8 = v40[-1].m128i_i64[1];
    if ( !v7 )
    {
LABEL_43:
      v22 = a1[1].m128i_i64[1];
      if ( v8 && v22 )
      {
        sub_15B1350((__int64)v43, *(unsigned __int64 **)(v22 + 24), *(unsigned __int64 **)(v22 + 32));
        sub_15B1350((__int64)v46, *(unsigned __int64 **)(v8 + 24), *(unsigned __int64 **)(v8 + 32));
        if ( v45 )
        {
          if ( !v48 )
            goto LABEL_63;
          v23 = v44 < v47;
        }
        else
        {
          v23 = v48;
        }
        if ( v23 )
        {
LABEL_48:
          v24 = _mm_loadu_si128(a1 + 1);
          v13 = a1->m128i_i64[1];
          a1[1].m128i_i64[0] = a1->m128i_i64[0];
          a1[1].m128i_i64[1] = v13;
          *a1 = v24;
          goto LABEL_14;
        }
      }
      else if ( v8 )
      {
        goto LABEL_48;
      }
LABEL_63:
      v33 = v4->m128i_i64[1];
      v34 = v40[-1].m128i_i64[1];
      if ( v33 && v34 )
      {
        sub_15B1350((__int64)v43, *(unsigned __int64 **)(v33 + 24), *(unsigned __int64 **)(v33 + 32));
        sub_15B1350((__int64)v46, *(unsigned __int64 **)(v34 + 24), *(unsigned __int64 **)(v34 + 32));
        if ( v45 )
        {
          if ( !v48 )
          {
            v35 = a1->m128i_i64[0];
            v36 = a1->m128i_i64[1];
LABEL_68:
            *a1 = _mm_loadu_si128(v4);
            v4->m128i_i64[0] = v35;
            v4->m128i_i64[1] = v36;
            v13 = a1[1].m128i_i64[1];
            goto LABEL_14;
          }
          v37 = v44 < v47;
        }
        else
        {
          v37 = v48;
        }
      }
      else
      {
        v37 = v34 != 0;
      }
      v35 = a1->m128i_i64[0];
      v36 = a1->m128i_i64[1];
      if ( v37 )
      {
        *a1 = _mm_loadu_si128(v40 - 1);
        v40[-1].m128i_i64[0] = v35;
        v40[-1].m128i_i64[1] = v36;
        v13 = a1[1].m128i_i64[1];
        goto LABEL_14;
      }
      goto LABEL_68;
    }
    v9 = v4->m128i_i64[1];
    if ( !v8 || !v9 )
    {
      if ( v8 )
        goto LABEL_13;
      goto LABEL_56;
    }
    sub_15B1350((__int64)v43, *(unsigned __int64 **)(v9 + 24), *(unsigned __int64 **)(v9 + 32));
    sub_15B1350((__int64)v46, *(unsigned __int64 **)(v8 + 24), *(unsigned __int64 **)(v8 + 32));
    if ( v45 )
    {
      if ( !v48 )
        goto LABEL_56;
      v10 = v44 < v47;
    }
    else
    {
      v10 = v48;
    }
    if ( !v10 )
    {
LABEL_56:
      v29 = a1[1].m128i_i64[1];
      v30 = v40[-1].m128i_i64[1];
      if ( v30 && v29 )
      {
        sub_15B1350((__int64)v43, *(unsigned __int64 **)(v29 + 24), *(unsigned __int64 **)(v29 + 32));
        sub_15B1350((__int64)v46, *(unsigned __int64 **)(v30 + 24), *(unsigned __int64 **)(v30 + 32));
        if ( v45 )
        {
          if ( !v48 )
          {
            v31 = a1->m128i_i64[0];
            v13 = a1->m128i_i64[1];
LABEL_61:
            v32 = _mm_loadu_si128(a1 + 1);
            a1[1].m128i_i64[0] = v31;
            a1[1].m128i_i64[1] = v13;
            *a1 = v32;
            goto LABEL_14;
          }
          v38 = v44 < v47;
        }
        else
        {
          v38 = v48;
        }
      }
      else
      {
        v38 = v30 != 0;
      }
      v31 = a1->m128i_i64[0];
      v13 = a1->m128i_i64[1];
      if ( v38 )
      {
        *a1 = _mm_loadu_si128(v40 - 1);
        v40[-1].m128i_i64[0] = v31;
        v40[-1].m128i_i64[1] = v13;
        v13 = a1[1].m128i_i64[1];
        goto LABEL_14;
      }
      goto LABEL_61;
    }
LABEL_13:
    v11 = a1->m128i_i64[0];
    v12 = a1->m128i_i64[1];
    *a1 = _mm_loadu_si128(v4);
    v4->m128i_i64[0] = v11;
    v4->m128i_i64[1] = v12;
    v13 = a1[1].m128i_i64[1];
LABEL_14:
    v14 = a1->m128i_i64[1];
    v15 = a1 + 1;
    v16 = v40;
    while ( 1 )
    {
      v42 = (__int64 *)v15;
      if ( v14 )
      {
        if ( v13 )
          break;
      }
      v17 = v14 != 0;
LABEL_21:
      if ( !v17 )
        goto LABEL_22;
LABEL_15:
      v13 = v15[1].m128i_i64[1];
      ++v15;
    }
    v41 = v14;
    sub_15B1350((__int64)v43, *(unsigned __int64 **)(v13 + 24), *(unsigned __int64 **)(v13 + 32));
    sub_15B1350((__int64)v46, *(unsigned __int64 **)(v41 + 24), *(unsigned __int64 **)(v41 + 32));
    if ( !v45 )
    {
      v17 = v48;
LABEL_20:
      v14 = a1->m128i_i64[1];
      goto LABEL_21;
    }
    if ( v48 )
    {
      v17 = v44 < v47;
      goto LABEL_20;
    }
    v14 = a1->m128i_i64[1];
LABEL_22:
    --v16;
    while ( 2 )
    {
      v19 = v16->m128i_i64[1];
      if ( !v14 || !v19 )
      {
        v18 = v19 != 0;
        goto LABEL_24;
      }
      sub_15B1350((__int64)v43, *(unsigned __int64 **)(v14 + 24), *(unsigned __int64 **)(v14 + 32));
      sub_15B1350((__int64)v46, *(unsigned __int64 **)(v19 + 24), *(unsigned __int64 **)(v19 + 32));
      if ( !v45 )
      {
        v18 = v48;
LABEL_24:
        if ( !v18 )
          goto LABEL_30;
        v14 = a1->m128i_i64[1];
        --v16;
        continue;
      }
      break;
    }
    if ( v48 )
    {
      v18 = v44 < v47;
      goto LABEL_24;
    }
LABEL_30:
    if ( v15 < v16 )
    {
      v20 = v15->m128i_i64[1];
      v21 = v15->m128i_i64[0];
      *v15 = _mm_loadu_si128(v16);
      v16->m128i_i64[0] = v21;
      v16->m128i_i64[1] = v20;
      v14 = a1->m128i_i64[1];
      goto LABEL_15;
    }
    sub_3988120(v15, v40, v39);
    result = (char *)v15 - (char *)a1;
    if ( (char *)v15 - (char *)a1 > 256 )
    {
      if ( v39 )
      {
        v40 = v15;
        continue;
      }
LABEL_51:
      v25 = result >> 4;
      v26 = ((result >> 4) - 2) >> 1;
      sub_3987010((__int64)a1, v26, result >> 4, a1[v26].m128i_i64[0], a1[v26].m128i_i64[1]);
      do
      {
        --v26;
        sub_3987010((__int64)a1, v26, v25, a1[v26].m128i_i64[0], a1[v26].m128i_i64[1]);
      }
      while ( v26 );
      do
      {
        v42 -= 2;
        v27 = *v42;
        v28 = v42[1];
        *(__m128i *)v42 = _mm_loadu_si128(a1);
        result = sub_3987010((__int64)a1, 0, ((char *)v42 - (char *)a1) >> 4, v27, v28);
      }
      while ( (char *)v42 - (char *)a1 > 16 );
    }
    return result;
  }
}
