// Function: sub_351BDD0
// Address: 0x351bdd0
//
__int64 *__fastcall sub_351BDD0(
        __int64 *a1,
        __int64 *a2,
        __int64 *a3,
        __int64 a4,
        __int64 a5,
        __int64 *a6,
        __int64 a7,
        __int128 a8)
{
  __int64 *v8; // r15
  __m128i v10; // xmm0
  __int64 v11; // r14
  __int64 v12; // rax
  __int64 *v13; // r13
  __int64 *v14; // r11
  __int64 v15; // rbx
  __int64 v16; // r14
  __int64 v17; // r13
  __int64 *v18; // rax
  __int64 *v19; // r11
  __int64 *v20; // r10
  char *v21; // rax
  int v22; // ecx
  __int64 v23; // rdx
  signed __int64 v24; // rbx
  __int64 *result; // rax
  void **v26; // rbx
  __int64 *v27; // r15
  void *v28; // rax
  __int64 v29; // rax
  __int64 *v30; // rdi
  __int64 *v31; // rsi
  size_t v32; // rdx
  __int64 *v33; // rbx
  __int64 *v34; // r15
  __int64 *v35; // r12
  __int64 v36; // r13
  __int64 *i; // r14
  unsigned int v38; // ebx
  __int64 *v39; // rax
  __int64 *v40; // [rsp+8h] [rbp-78h]
  int v41; // [rsp+8h] [rbp-78h]
  __int64 *v42; // [rsp+10h] [rbp-70h]
  __int64 v43; // [rsp+18h] [rbp-68h]
  __int64 *v44; // [rsp+18h] [rbp-68h]
  char *v45; // [rsp+28h] [rbp-58h]
  __int64 *v46; // [rsp+28h] [rbp-58h]
  unsigned int v47; // [rsp+30h] [rbp-50h]
  __int64 v48; // [rsp+30h] [rbp-50h]
  void *srca; // [rsp+40h] [rbp-40h]

  v8 = a2;
  v10 = _mm_loadu_si128((const __m128i *)&a8);
  v11 = v10.m128i_i64[0];
  v12 = a5;
  v13 = (__int64 *)v10.m128i_i64[1];
  if ( a7 <= a5 )
    v12 = a7;
  if ( a4 <= v12 )
    goto LABEL_12;
  if ( a7 >= a5 )
    goto LABEL_28;
  v14 = a2;
  v15 = a4;
  v16 = a5;
  while ( 1 )
  {
    v40 = v14;
    if ( v16 < v15 )
    {
      v43 = v15 / 2;
      v39 = sub_3511A00(v14, (__int64)a3, &a1[v15 / 2], v10.m128i_i64[0], (__int64 *)v10.m128i_i64[1]);
      v19 = v40;
      v20 = &a1[v15 / 2];
      v42 = v39;
      v17 = v39 - v40;
    }
    else
    {
      v17 = v16 / 2;
      v42 = &v14[v16 / 2];
      v18 = sub_3511940(a1, (__int64)v14, v42, v10.m128i_i64[0], (__int64 *)v10.m128i_i64[1]);
      v19 = v40;
      v20 = v18;
      v43 = v18 - a1;
    }
    v15 -= v43;
    v41 = (int)v20;
    v16 -= v17;
    v21 = sub_351BCC0(v20, v19, v42, v15, v17, a6, a7);
    v22 = v43;
    v44 = (__int64 *)v21;
    sub_351BDD0((_DWORD)a1, v41, (_DWORD)v21, v22, v17, (_DWORD)a6, a7, *(_OWORD *)&v10);
    v23 = a7;
    if ( v16 <= a7 )
      v23 = v16;
    if ( v15 <= v23 )
    {
      v13 = (__int64 *)v10.m128i_i64[1];
      v11 = v10.m128i_i64[0];
      a1 = v44;
      v8 = v42;
LABEL_12:
      v24 = (char *)v8 - (char *)a1;
      if ( a1 != v8 )
        memmove(a6, a1, (char *)v8 - (char *)a1);
      result = (__int64 *)((char *)a6 + v24);
      v45 = (char *)a6 + v24;
      if ( a6 == (__int64 *)((char *)a6 + v24) )
        return result;
      if ( a3 == v8 )
        goto LABEL_22;
      v26 = (void **)v8;
      v27 = a1;
      while ( 1 )
      {
        srca = *v26;
        v47 = sub_2E441D0(*(_QWORD *)(v11 + 528), *v13, *a6);
        if ( v47 < (unsigned int)sub_2E441D0(*(_QWORD *)(v11 + 528), *v13, (__int64)srca) )
        {
          v28 = *v26;
          ++v27;
          ++v26;
          *(v27 - 1) = (__int64)v28;
          if ( v45 == (char *)a6 )
            goto LABEL_21;
        }
        else
        {
          v29 = *a6;
          ++v27;
          ++a6;
          *(v27 - 1) = v29;
          if ( v45 == (char *)a6 )
          {
LABEL_21:
            a1 = v27;
LABEL_22:
            result = (__int64 *)v45;
            if ( a6 != (__int64 *)v45 )
            {
              v30 = a1;
              v31 = a6;
              v32 = v45 - (char *)a6;
              return (__int64 *)memmove(v30, v31, v32);
            }
            return result;
          }
        }
        if ( a3 == (__int64 *)v26 )
          goto LABEL_21;
      }
    }
    if ( v16 <= a7 )
      break;
    a1 = v44;
    v14 = v42;
  }
  v13 = (__int64 *)v10.m128i_i64[1];
  v11 = v10.m128i_i64[0];
  a1 = v44;
  v8 = v42;
LABEL_28:
  if ( a3 != v8 )
    memmove(a6, v8, (char *)a3 - (char *)v8);
  result = (__int64 *)((char *)a6 + (char *)a3 - (char *)v8);
  if ( a1 == v8 )
  {
    if ( a6 == result )
      return result;
    v32 = (char *)a3 - (char *)v8;
    v30 = v8;
LABEL_41:
    v31 = a6;
    return (__int64 *)memmove(v30, v31, v32);
  }
  else
  {
    if ( a6 == result )
      return result;
    v33 = v8 - 1;
    v46 = a6;
    v34 = result - 1;
    v35 = v13;
    v36 = v11;
    for ( i = v33; ; --i )
    {
      while ( 1 )
      {
        v48 = *v34;
        v38 = sub_2E441D0(*(_QWORD *)(v36 + 528), *v35, *i);
        --a3;
        if ( v38 < (unsigned int)sub_2E441D0(*(_QWORD *)(v36 + 528), *v35, v48) )
          break;
        result = (__int64 *)*v34;
        *a3 = *v34;
        if ( v46 == v34 )
          return result;
        --v34;
      }
      result = (__int64 *)*i;
      *a3 = *i;
      if ( i == a1 )
        break;
    }
    a6 = v46;
    if ( v46 != v34 + 1 )
    {
      v32 = (char *)(v34 + 1) - (char *)v46;
      v30 = (__int64 *)((char *)a3 - v32);
      goto LABEL_41;
    }
  }
  return result;
}
