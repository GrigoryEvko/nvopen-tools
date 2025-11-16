// Function: sub_25946B0
// Address: 0x25946b0
//
__int64 __fastcall sub_25946B0(__int64 a1, _QWORD *a2)
{
  unsigned __int64 *v3; // rdi
  unsigned __int64 *v4; // r14
  __m128i v5; // rax
  unsigned int v6; // r12d
  char v7; // al
  __int64 v8; // rdx
  __m128i v10; // rax
  __m128i v11; // rax
  __int64 v12; // r15
  unsigned __int64 *v13; // r14
  __int64 v14; // rax
  __int64 v15; // r15
  unsigned __int64 *v16; // rax
  char *v17; // r15
  char *v18; // r15
  __m128i v19; // rax
  char *v20; // r15
  __m128i v21; // rax
  char *v22; // r15
  __m128i v23; // rax
  __m128i v24; // rax
  char *v25; // r12
  __m128i v26; // rax
  char v27; // al
  unsigned __int64 *v28; // [rsp+0h] [rbp-D0h]
  unsigned __int64 *v29; // [rsp+8h] [rbp-C8h]
  unsigned __int64 v30; // [rsp+18h] [rbp-B8h]
  unsigned __int64 *v31; // [rsp+18h] [rbp-B8h]
  char v32; // [rsp+2Eh] [rbp-A2h] BYREF
  char v33; // [rsp+2Fh] [rbp-A1h] BYREF
  __m128i v34; // [rsp+30h] [rbp-A0h] BYREF
  __m128i v35; // [rsp+40h] [rbp-90h] BYREF
  char *v36; // [rsp+50h] [rbp-80h]
  unsigned __int64 *v37; // [rsp+60h] [rbp-70h] BYREF
  __int64 v38; // [rsp+68h] [rbp-68h]
  _BYTE v39[96]; // [rsp+70h] [rbp-60h] BYREF

  v32 = 0;
  v30 = sub_250D070((_QWORD *)(a1 + 72));
  v37 = (unsigned __int64 *)v39;
  v38 = 0x300000000LL;
  if ( !(unsigned __int8)sub_2526B50((__int64)a2, (const __m128i *)(a1 + 72), a1, (__int64)&v37, 3u, &v32, 1u)
    || (v3 = v37, (unsigned int)v38 == 1) && v30 == *v37 )
  {
    v7 = *(_BYTE *)v30;
    if ( *(_BYTE *)v30 != 84 )
      goto LABEL_10;
    v12 = 32LL * (*(_DWORD *)(v30 + 4) & 0x7FFFFFF);
    if ( (*(_BYTE *)(v30 + 7) & 0x40) != 0 )
    {
      v13 = *(unsigned __int64 **)(v30 - 8);
      v28 = &v13[(unsigned __int64)v12 / 8];
    }
    else
    {
      v28 = (unsigned __int64 *)v30;
      v13 = (unsigned __int64 *)(v30 - v12);
    }
    v35.m128i_i64[0] = (__int64)a2;
    v14 = v12 >> 5;
    v15 = v12 >> 7;
    v35.m128i_i64[1] = a1;
    v36 = &v33;
    if ( v15 )
    {
      v16 = &v13[16 * v15];
      v17 = &v33;
      v29 = v16;
      while ( 1 )
      {
        v24.m128i_i64[0] = sub_250D2C0(*v13, 0);
        v34 = v24;
        if ( !(unsigned __int8)sub_258F340(v35.m128i_i64[0], v35.m128i_i64[1], &v34, 1, v17, 0, 0) )
          goto LABEL_33;
        v18 = v36;
        v19.m128i_i64[0] = sub_250D2C0(v13[4], 0);
        v34 = v19;
        if ( !(unsigned __int8)sub_258F340(v35.m128i_i64[0], v35.m128i_i64[1], &v34, 1, v18, 0, 0) )
        {
          v13 += 4;
          goto LABEL_33;
        }
        v20 = v36;
        v21.m128i_i64[0] = sub_250D2C0(v13[8], 0);
        v34 = v21;
        if ( !(unsigned __int8)sub_258F340(v35.m128i_i64[0], v35.m128i_i64[1], &v34, 1, v20, 0, 0) )
        {
          v13 += 8;
          goto LABEL_33;
        }
        v22 = v36;
        v23.m128i_i64[0] = sub_250D2C0(v13[12], 0);
        v34 = v23;
        if ( !(unsigned __int8)sub_258F340(v35.m128i_i64[0], v35.m128i_i64[1], &v34, 1, v22, 0, 0) )
        {
          v13 += 12;
          goto LABEL_33;
        }
        v13 += 16;
        if ( v13 == v29 )
        {
          v14 = ((char *)v28 - (char *)v13) >> 5;
          break;
        }
        v17 = v36;
      }
    }
    if ( v14 != 2 )
    {
      if ( v14 != 3 )
      {
        if ( v14 != 1 )
          goto LABEL_22;
        goto LABEL_41;
      }
      if ( !(unsigned __int8)sub_258F3F0((__int64)&v35, *v13) )
      {
LABEL_33:
        if ( v13 == v28 )
          goto LABEL_22;
        v7 = *(_BYTE *)v30;
LABEL_10:
        if ( v7 != 86
          || (v10.m128i_i64[0] = sub_250D2C0(*(_QWORD *)(v30 - 32), 0),
              v34 = v10,
              !(unsigned __int8)sub_258F340(a2, a1, &v34, 1, &v33, 0, 0))
          || (v11.m128i_i64[0] = sub_250D2C0(*(_QWORD *)(v30 - 64), 0),
              v35 = v11,
              !(unsigned __int8)sub_258F340(a2, a1, &v35, 1, &v33, 0, 0)) )
        {
          if ( (v35.m128i_i64[0] = sub_250D2C0(v30, 0), v35.m128i_i64[1] = v8, v35.m128i_i64[0] == *(_QWORD *)(a1 + 72))
            && *(_QWORD *)(a1 + 80) == v8
            || !(unsigned __int8)sub_258F340(a2, a1, &v35, 1, &v34, 0, 0) )
          {
            v6 = 0;
            *(_BYTE *)(a1 + 97) = *(_BYTE *)(a1 + 96);
LABEL_14:
            v3 = v37;
            goto LABEL_15;
          }
        }
LABEL_22:
        v6 = 1;
        goto LABEL_14;
      }
      v13 += 4;
    }
    if ( (unsigned __int8)sub_258F3F0((__int64)&v35, *v13) )
    {
      v13 += 4;
LABEL_41:
      v25 = v36;
      v26.m128i_i64[0] = sub_250D2C0(*v13, 0);
      v34 = v26;
      v27 = sub_258F340(v35.m128i_i64[0], v35.m128i_i64[1], &v34, 1, v25, 0, 0);
      v6 = 1;
      if ( v27 )
        goto LABEL_14;
      goto LABEL_33;
    }
    goto LABEL_33;
  }
  v31 = &v37[2 * (unsigned int)v38];
  if ( v31 == v37 )
  {
    v6 = 1;
  }
  else
  {
    v4 = v37;
    do
    {
      v5.m128i_i64[0] = sub_250D2C0(*v4, 0);
      v35 = v5;
      if ( !(unsigned __int8)sub_258F340(a2, a1, &v35, 1, &v34, 0, 0) )
      {
        v3 = v37;
        v6 = 0;
        *(_BYTE *)(a1 + 97) = *(_BYTE *)(a1 + 96);
        goto LABEL_15;
      }
      v4 += 2;
    }
    while ( v31 != v4 );
    v3 = v37;
    v6 = 1;
  }
LABEL_15:
  if ( v3 != (unsigned __int64 *)v39 )
    _libc_free((unsigned __int64)v3);
  return v6;
}
