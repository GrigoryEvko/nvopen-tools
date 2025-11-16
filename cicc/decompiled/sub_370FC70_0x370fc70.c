// Function: sub_370FC70
// Address: 0x370fc70
//
__int64 *__fastcall sub_370FC70(__int64 *a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  _QWORD *v4; // r15
  unsigned __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rsi
  unsigned int v10; // edx
  __int64 v11; // rcx
  unsigned __int64 v12; // rax
  unsigned int v14; // r8d
  __int64 v15; // rdi
  __int64 v16; // rdx
  __int64 v17; // rax
  unsigned int v18; // r8d
  unsigned __int64 v19; // rax
  unsigned __int64 v20; // rax
  __int64 v21; // rax
  _QWORD *v22; // rbx
  __m128i *v23; // rsi
  __int64 v24; // rdi
  __int64 v25; // rax
  char v26; // al
  __int64 v27; // [rsp+10h] [rbp-90h]
  __int64 v28; // [rsp+10h] [rbp-90h]
  unsigned __int64 v30; // [rsp+18h] [rbp-88h]
  _QWORD *v31; // [rsp+18h] [rbp-88h]
  unsigned int v32; // [rsp+24h] [rbp-7Ch] BYREF
  __int64 v33; // [rsp+28h] [rbp-78h] BYREF
  __m128i v34; // [rsp+30h] [rbp-70h] BYREF
  __m128i v35[2]; // [rsp+40h] [rbp-60h] BYREF
  __int16 v36; // [rsp+60h] [rbp-40h]

  v4 = a2 + 2;
  v35[0].m128i_i64[0] = (__int64)"CompleteClass";
  v36 = 259;
  sub_37011E0((unsigned __int64 *)&v34, a2 + 2, (unsigned int *)(a4 + 2), v35[0].m128i_i64);
  if ( (v34.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *a1 = v34.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL | 1;
    return a1;
  }
  v35[0].m128i_i64[0] = (__int64)"OverriddenVFTable";
  v36 = 259;
  sub_37011E0((unsigned __int64 *)&v34, v4, (unsigned int *)(a4 + 6), v35[0].m128i_i64);
  v7 = v34.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL;
  if ( (v34.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL) != 0
    || (v34.m128i_i64[0] = 0,
        sub_9C66B0(v34.m128i_i64),
        v35[0].m128i_i64[0] = (__int64)"VFPtrOffset",
        v36 = 259,
        sub_370BDF0((unsigned __int64 *)&v34, v4, (unsigned int *)(a4 + 12), v35),
        v7 = v34.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL,
        (v34.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL) != 0) )
  {
    v34.m128i_i64[0] = 0;
    *a1 = v7 | 1;
    sub_9C66B0(v34.m128i_i64);
    return a1;
  }
  v34.m128i_i64[0] = 0;
  sub_9C66B0(v34.m128i_i64);
  v32 = 0;
  if ( !a2[7] || a2[9] || a2[8] )
  {
    v8 = *(_QWORD *)(a4 + 16);
    v9 = *(_QWORD *)(a4 + 24);
    if ( v9 != v8 )
    {
      v10 = 0;
      do
      {
        v11 = *(_QWORD *)(v8 + 8);
        v8 += 16;
        v10 += v11 + 1;
      }
      while ( v9 != v8 );
      v32 = v10;
    }
  }
  v36 = 257;
  sub_370BDF0((unsigned __int64 *)&v34, v4, &v32, v35);
  v12 = v34.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL;
  if ( (v34.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL) == 0 )
  {
    v34.m128i_i64[0] = 0;
    sub_9C66B0(v34.m128i_i64);
    v15 = a2[9];
    if ( v15 )
    {
      v16 = a2[7];
      if ( v16 )
        goto LABEL_17;
      if ( a2[8] )
      {
LABEL_46:
        if ( !a2[8] )
        {
LABEL_30:
          v31 = *(_QWORD **)(a4 + 24);
          if ( *(_QWORD **)(a4 + 16) != v31 )
          {
            v22 = *(_QWORD **)(a4 + 16);
            while ( 1 )
            {
              v35[0].m128i_i64[0] = (__int64)"MethodName";
              v36 = 259;
              sub_3701560((unsigned __int64 *)&v34, v4, v22, v35, v14);
              if ( (v34.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
                break;
              v22 += 2;
              if ( v31 == v22 )
                goto LABEL_34;
            }
            v35[0].m128i_i64[0] = 0;
            v34.m128i_i64[0] = v34.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL | 1;
            sub_9C6670(v35[0].m128i_i64, &v34);
            sub_9C66B0(v34.m128i_i64);
            v20 = v35[0].m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL;
            if ( (v35[0].m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
            {
LABEL_24:
              *a1 = 0;
              v35[0].m128i_i64[0] = v20 | 1;
              sub_9C6670(a1, v35);
              sub_9C66B0(v35[0].m128i_i64);
              return a1;
            }
          }
LABEL_34:
          v35[0].m128i_i64[0] = 0;
          sub_9C66B0(v35[0].m128i_i64);
          *a1 = 1;
          sub_9C66B0(v35[0].m128i_i64);
          return a1;
        }
LABEL_17:
        v34 = 0u;
        while ( 1 )
        {
          if ( *(_BYTE *)(v16 + 48) )
          {
            v17 = *(_QWORD *)(v16 + 40);
          }
          else
          {
            v24 = *(_QWORD *)(v16 + 24);
            v17 = 0;
            if ( v24 )
            {
              v27 = v16;
              v25 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v24 + 40LL))(v24);
              v16 = v27;
              v17 = v25 - *(_QWORD *)(v27 + 32);
            }
          }
          if ( *(_QWORD *)(v16 + 56) == v17 || (unsigned __int8)sub_1254BC0(a2[7]) > 0xEFu )
            goto LABEL_34;
          v35[0].m128i_i64[0] = (__int64)"MethodName";
          v36 = 259;
          sub_3701560((unsigned __int64 *)&v33, v4, &v34, v35, v18);
          v19 = v33 & 0xFFFFFFFFFFFFFFFELL;
          if ( (v33 & 0xFFFFFFFFFFFFFFFELL) != 0 )
          {
            v30 = v33 & 0xFFFFFFFFFFFFFFFELL;
            v33 = 0;
            v35[0].m128i_i64[0] = v19 | 1;
            sub_9C66B0(&v33);
            v20 = v30;
            goto LABEL_24;
          }
          v23 = *(__m128i **)(a4 + 24);
          if ( v23 == *(__m128i **)(a4 + 32) )
          {
            sub_A04210((const __m128i **)(a4 + 16), v23, &v34);
          }
          else
          {
            if ( v23 )
            {
              *v23 = _mm_loadu_si128(&v34);
              v23 = *(__m128i **)(a4 + 24);
            }
            *(_QWORD *)(a4 + 24) = v23 + 1;
          }
          v16 = a2[7];
        }
      }
      v28 = a2[7];
      v26 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v15 + 40LL))(v15);
      v16 = v28;
      if ( v26 )
      {
        v35[0].m128i_i64[0] = (__int64)"VFTableName";
        v36 = 259;
        (*(void (__fastcall **)(_QWORD, __m128i *, __int64))(*(_QWORD *)a2[9] + 24LL))(a2[9], v35, v28);
        v16 = v28;
      }
      v21 = a2[7];
      if ( a2[9] )
      {
        if ( v21 )
        {
          v16 = a2[7];
          goto LABEL_17;
        }
        goto LABEL_46;
      }
    }
    else
    {
      v21 = a2[7];
    }
    v16 = v21;
    if ( a2[8] && !v21 )
      goto LABEL_30;
    goto LABEL_17;
  }
  *a1 = 0;
  v34.m128i_i64[0] = v12 | 1;
  sub_9C6670(a1, &v34);
  sub_9C66B0(v34.m128i_i64);
  return a1;
}
