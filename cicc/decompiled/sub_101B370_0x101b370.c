// Function: sub_101B370
// Address: 0x101b370
//
unsigned __int8 *__fastcall sub_101B370(int a1, __int64 *a2, __int64 *a3, __m128i *a4, int a5)
{
  unsigned int v5; // r15d
  int v9; // eax
  int v10; // edx
  __int64 *v11; // rax
  __int64 *v12; // rsi
  __int64 *v13; // r14
  __int64 *v14; // rdx
  unsigned __int8 *result; // rax
  __int64 *v16; // r10
  __int64 *v17; // rax
  __int64 *v18; // r14
  __int64 *v19; // rax
  __int64 *v20; // rax
  __int64 *v21; // rax
  __int64 *v22; // [rsp-58h] [rbp-58h]
  __int64 *v23; // [rsp-50h] [rbp-50h]
  __int64 *v24; // [rsp-50h] [rbp-50h]
  __int64 *v25; // [rsp-48h] [rbp-48h]
  __int64 *v26; // [rsp-48h] [rbp-48h]
  __int64 *v27; // [rsp-40h] [rbp-40h]
  __int64 *v28; // [rsp-40h] [rbp-40h]

  if ( !a5 )
    return 0;
  v5 = a5 - 1;
  v9 = *(unsigned __int8 *)a2;
  v10 = *(unsigned __int8 *)a3;
  if ( (unsigned __int8)v9 <= 0x1Cu || (unsigned int)(v9 - 42) > 0x11 )
  {
    if ( (unsigned __int8)v10 <= 0x1Cu || (unsigned int)(v10 - 42) > 0x11 )
      return 0;
    v16 = 0;
    goto LABEL_29;
  }
  if ( (unsigned __int8)v10 > 0x1Cu )
  {
    if ( (unsigned int)(v10 - 42) > 0x11 )
    {
      if ( a1 != v9 - 29 )
        goto LABEL_23;
      v12 = (__int64 *)*(a2 - 4);
      v27 = (__int64 *)*(a2 - 8);
      v14 = (__int64 *)sub_101AFF0(a1, v12, a3, a4, v5);
      if ( !v14 )
        goto LABEL_23;
      v13 = 0;
      goto LABEL_7;
    }
    v16 = a2;
    if ( a1 == v9 - 29 )
    {
      v12 = (__int64 *)*(a2 - 4);
      v13 = a3;
      v27 = (__int64 *)*(a2 - 8);
      v14 = (__int64 *)sub_101AFF0(a1, v12, a3, a4, v5);
      if ( !v14 )
        goto LABEL_10;
      goto LABEL_7;
    }
LABEL_29:
    v13 = a3;
    if ( a1 != *(unsigned __int8 *)a3 - 29 )
      goto LABEL_11;
    goto LABEL_30;
  }
  if ( a1 != v9 - 29 )
    return 0;
  v11 = (__int64 *)*(a2 - 8);
  v12 = (__int64 *)*(a2 - 4);
  v13 = 0;
  v27 = v11;
  v14 = (__int64 *)sub_101AFF0(a1, v12, a3, a4, v5);
  if ( !v14 )
  {
LABEL_23:
    if ( ((1LL << a1) & 0x70066000) != 0 )
    {
      v16 = a2;
      v13 = 0;
      goto LABEL_13;
    }
    return 0;
  }
LABEL_7:
  if ( v14 == v12 )
    return (unsigned __int8 *)a2;
  result = sub_101AFF0(a1, v27, v14, a4, v5);
  if ( result )
    return result;
  if ( !v13 )
    goto LABEL_23;
LABEL_10:
  v16 = a2;
  if ( a1 != *(unsigned __int8 *)v13 - 29 )
    goto LABEL_11;
LABEL_30:
  v25 = v16;
  v22 = (__int64 *)*(v13 - 4);
  v23 = (__int64 *)*(v13 - 8);
  v20 = (__int64 *)sub_101AFF0(a1, a2, v23, a4, v5);
  v16 = v25;
  if ( v20 )
  {
    if ( v20 == v23 )
      return (unsigned __int8 *)a3;
    result = sub_101AFF0(a1, v20, v22, a4, v5);
    v16 = v25;
    if ( result )
      return result;
  }
LABEL_11:
  if ( ((1LL << a1) & 0x70066000) == 0 )
    return 0;
  if ( !v16 )
    goto LABEL_14;
LABEL_13:
  if ( a1 != *(unsigned __int8 *)v16 - 29
    || (v24 = (__int64 *)*(v16 - 4),
        v26 = (__int64 *)*(v16 - 8),
        (v21 = (__int64 *)sub_101AFF0(a1, a3, v26, a4, v5)) == 0) )
  {
LABEL_14:
    if ( v13 )
    {
      if ( a1 == *(unsigned __int8 *)v13 - 29 )
      {
        v17 = (__int64 *)*(v13 - 8);
        v18 = (__int64 *)*(v13 - 4);
        v28 = v17;
        v19 = (__int64 *)sub_101AFF0(a1, v18, a2, a4, v5);
        if ( v19 )
        {
          if ( v19 != v18 )
            return sub_101AFF0(a1, v28, v19, a4, v5);
          return (unsigned __int8 *)a3;
        }
      }
    }
    return 0;
  }
  if ( v21 == v26 )
    return (unsigned __int8 *)a2;
  result = sub_101AFF0(a1, v21, v24, a4, v5);
  if ( !result )
    goto LABEL_14;
  return result;
}
