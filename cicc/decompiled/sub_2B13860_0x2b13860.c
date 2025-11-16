// Function: sub_2B13860
// Address: 0x2b13860
//
__int64 *__fastcall sub_2B13860(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r15
  __int64 v4; // rax
  __int64 *v6; // rbx
  __int64 *v7; // r15
  _BYTE *v8; // rdi
  unsigned __int64 v9; // rax
  _BYTE *v10; // rdi
  unsigned __int64 v11; // rax
  _BYTE *v12; // rdi
  unsigned __int64 v13; // rax
  _BYTE *v14; // rdi
  unsigned __int64 v15; // rax
  __int64 *result; // rax
  __int64 v17; // rdi
  unsigned __int64 v18; // rax
  __int64 v19; // rdi
  unsigned __int64 v20; // rax
  __int64 v21; // rdi
  unsigned __int64 v22; // rax
  char v23; // r8
  __m128i v24; // [rsp+0h] [rbp-80h] BYREF
  __int64 v25; // [rsp+10h] [rbp-70h]
  __int64 v26; // [rsp+18h] [rbp-68h]
  __int64 v27; // [rsp+20h] [rbp-60h]
  __int64 v28; // [rsp+28h] [rbp-58h]
  __int64 v29; // [rsp+30h] [rbp-50h]
  __int64 v30; // [rsp+38h] [rbp-48h]
  __int16 v31; // [rsp+40h] [rbp-40h]

  v3 = (a2 - (__int64)a1) >> 5;
  v4 = (a2 - (__int64)a1) >> 3;
  v6 = a1;
  if ( v3 > 0 )
  {
    v7 = &a1[4 * v3];
    while ( 1 )
    {
      v14 = (_BYTE *)*v6;
      if ( *(_BYTE *)*v6 != 13 )
      {
        v15 = *(_QWORD *)(*(_QWORD *)(a3 + 120) + 3344LL);
        v25 = 0;
        v24 = (__m128i)v15;
        v26 = 0;
        v27 = 0;
        v28 = 0;
        v29 = 0;
        v30 = 0;
        v31 = 257;
        if ( !(unsigned __int8)sub_9AC470((__int64)v14, &v24, 0) )
          return v6;
      }
      v8 = (_BYTE *)v6[1];
      if ( *v8 != 13 )
      {
        v9 = *(_QWORD *)(*(_QWORD *)(a3 + 120) + 3344LL);
        v25 = 0;
        v24 = (__m128i)v9;
        v26 = 0;
        v27 = 0;
        v28 = 0;
        v29 = 0;
        v30 = 0;
        v31 = 257;
        if ( !(unsigned __int8)sub_9AC470((__int64)v8, &v24, 0) )
          return v6 + 1;
      }
      v10 = (_BYTE *)v6[2];
      if ( *v10 != 13 )
      {
        v11 = *(_QWORD *)(*(_QWORD *)(a3 + 120) + 3344LL);
        v25 = 0;
        v24 = (__m128i)v11;
        v26 = 0;
        v27 = 0;
        v28 = 0;
        v29 = 0;
        v30 = 0;
        v31 = 257;
        if ( !(unsigned __int8)sub_9AC470((__int64)v10, &v24, 0) )
          return v6 + 2;
      }
      v12 = (_BYTE *)v6[3];
      if ( *v12 != 13 )
      {
        v13 = *(_QWORD *)(*(_QWORD *)(a3 + 120) + 3344LL);
        v31 = 257;
        v24 = (__m128i)v13;
        v25 = 0;
        v26 = 0;
        v27 = 0;
        v28 = 0;
        v29 = 0;
        v30 = 0;
        if ( !(unsigned __int8)sub_9AC470((__int64)v12, &v24, 0) )
          return v6 + 3;
      }
      v6 += 4;
      if ( v7 == v6 )
      {
        v4 = (a2 - (__int64)v6) >> 3;
        break;
      }
    }
  }
  if ( v4 == 2 )
    goto LABEL_25;
  if ( v4 == 3 )
  {
    v17 = *v6;
    if ( *(_BYTE *)*v6 != 13 )
    {
      v18 = *(_QWORD *)(*(_QWORD *)(a3 + 120) + 3344LL);
      v25 = 0;
      v24 = (__m128i)v18;
      v26 = 0;
      v27 = 0;
      v28 = 0;
      v29 = 0;
      v30 = 0;
      v31 = 257;
      if ( !(unsigned __int8)sub_9AC470(v17, &v24, 0) )
        return v6;
    }
    ++v6;
LABEL_25:
    v19 = *v6;
    if ( *(_BYTE *)*v6 == 13 )
      goto LABEL_27;
    v20 = *(_QWORD *)(*(_QWORD *)(a3 + 120) + 3344LL);
    v31 = 257;
    v24 = (__m128i)v20;
    v25 = 0;
    v26 = 0;
    v27 = 0;
    v28 = 0;
    v29 = 0;
    v30 = 0;
    if ( (unsigned __int8)sub_9AC470(v19, &v24, 0) )
    {
LABEL_27:
      ++v6;
      goto LABEL_28;
    }
    return v6;
  }
  if ( v4 != 1 )
    return (__int64 *)a2;
LABEL_28:
  v21 = *v6;
  if ( *(_BYTE *)*v6 == 13 )
    return (__int64 *)a2;
  v22 = *(_QWORD *)(*(_QWORD *)(a3 + 120) + 3344LL);
  v25 = 0;
  v24 = (__m128i)v22;
  v26 = 0;
  v27 = 0;
  v28 = 0;
  v29 = 0;
  v30 = 0;
  v31 = 257;
  v23 = sub_9AC470(v21, &v24, 0);
  result = v6;
  if ( v23 )
    return (__int64 *)a2;
  return result;
}
