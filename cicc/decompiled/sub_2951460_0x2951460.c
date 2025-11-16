// Function: sub_2951460
// Address: 0x2951460
//
__int64 *__fastcall sub_2951460(__int64 *a1, __int64 a2, unsigned __int64 *a3)
{
  __int64 v3; // r14
  __int64 v5; // rax
  __int64 *v6; // rbx
  __int64 *v7; // r14
  __int64 v8; // rdi
  __int64 v9; // rdi
  unsigned __int64 v10; // rax
  __int64 v11; // rdi
  __int64 v12; // rdi
  __int64 *result; // rax
  __int64 v14; // rdi
  char v15; // r8
  unsigned __int64 v16; // rax
  __int64 v17; // rdi
  char v18; // r8
  __int64 v19; // rdi
  bool v20; // zf
  __m128i v21; // [rsp+0h] [rbp-80h] BYREF
  __int64 v22; // [rsp+10h] [rbp-70h]
  __int64 v23; // [rsp+18h] [rbp-68h]
  __int64 v24; // [rsp+20h] [rbp-60h]
  __int64 v25; // [rsp+28h] [rbp-58h]
  __int64 v26; // [rsp+30h] [rbp-50h]
  __int64 v27; // [rsp+38h] [rbp-48h]
  __int16 v28; // [rsp+40h] [rbp-40h]

  v3 = (a2 - (__int64)a1) >> 7;
  v5 = (a2 - (__int64)a1) >> 5;
  v6 = a1;
  if ( v3 <= 0 )
  {
LABEL_11:
    if ( v5 != 2 )
    {
      if ( v5 != 3 )
      {
        if ( v5 != 1 )
          return (__int64 *)a2;
LABEL_19:
        v19 = *v6;
        v21 = (__m128i)*a3;
        v22 = 0;
        v23 = 0;
        v24 = 0;
        v25 = 0;
        v26 = 0;
        v27 = 0;
        v28 = 257;
        v20 = (unsigned __int8)sub_9AC470(v19, &v21, 0) == 0;
        result = v6;
        if ( !v20 )
          return (__int64 *)a2;
        return result;
      }
      v14 = *v6;
      v21 = (__m128i)*a3;
      v22 = 0;
      v23 = 0;
      v24 = 0;
      v25 = 0;
      v26 = 0;
      v27 = 0;
      v28 = 257;
      v15 = sub_9AC470(v14, &v21, 0);
      result = v6;
      if ( !v15 )
        return result;
      v6 += 4;
    }
    v16 = *a3;
    v17 = *v6;
    v28 = 257;
    v21 = (__m128i)v16;
    v22 = 0;
    v23 = 0;
    v24 = 0;
    v25 = 0;
    v26 = 0;
    v27 = 0;
    v18 = sub_9AC470(v17, &v21, 0);
    result = v6;
    if ( !v18 )
      return result;
    v6 += 4;
    goto LABEL_19;
  }
  v7 = &a1[16 * v3];
  while ( 1 )
  {
    v12 = *v6;
    v21 = (__m128i)*a3;
    v22 = 0;
    v23 = 0;
    v24 = 0;
    v25 = 0;
    v26 = 0;
    v27 = 0;
    v28 = 257;
    if ( !(unsigned __int8)sub_9AC470(v12, &v21, 0) )
      return v6;
    v8 = v6[4];
    v21 = (__m128i)*a3;
    v22 = 0;
    v23 = 0;
    v24 = 0;
    v25 = 0;
    v26 = 0;
    v27 = 0;
    v28 = 257;
    if ( !(unsigned __int8)sub_9AC470(v8, &v21, 0) )
      return v6 + 4;
    v9 = v6[8];
    v21 = (__m128i)*a3;
    v22 = 0;
    v23 = 0;
    v24 = 0;
    v25 = 0;
    v26 = 0;
    v27 = 0;
    v28 = 257;
    if ( !(unsigned __int8)sub_9AC470(v9, &v21, 0) )
      return v6 + 8;
    v10 = *a3;
    v11 = v6[12];
    v28 = 257;
    v21 = (__m128i)v10;
    v22 = 0;
    v23 = 0;
    v24 = 0;
    v25 = 0;
    v26 = 0;
    v27 = 0;
    if ( !(unsigned __int8)sub_9AC470(v11, &v21, 0) )
      return v6 + 12;
    v6 += 16;
    if ( v7 == v6 )
    {
      v5 = (a2 - (__int64)v6) >> 5;
      goto LABEL_11;
    }
  }
}
