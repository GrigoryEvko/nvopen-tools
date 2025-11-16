// Function: sub_2352D90
// Address: 0x2352d90
//
__int64 __fastcall sub_2352D90(__int64 a1, _BYTE *a2, unsigned __int64 a3)
{
  __int64 v3; // rdx
  _QWORD *i; // rax
  unsigned __int64 *v5; // r15
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // r12
  __m128i *v8; // rsi
  __int64 v9; // r9
  unsigned __int64 v10; // rdx
  unsigned __int64 v11; // rsi
  char v12; // di
  _BYTE *v13; // rax
  _BYTE *v14; // rdi
  __int64 v15; // rsi
  _BYTE *v16; // r8
  int v17; // edx
  unsigned int v18; // ecx
  unsigned __int64 v20; // rcx
  __int64 v21; // rax
  unsigned __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // [rsp+0h] [rbp-E0h]
  _BYTE *v26; // [rsp+20h] [rbp-C0h] BYREF
  unsigned __int64 v27; // [rsp+28h] [rbp-B8h]
  unsigned __int64 v28; // [rsp+30h] [rbp-B0h] BYREF
  __int64 v29; // [rsp+38h] [rbp-A8h]
  __int64 v30; // [rsp+40h] [rbp-A0h]
  __m128i v31; // [rsp+50h] [rbp-90h] BYREF
  __int64 v32; // [rsp+60h] [rbp-80h] BYREF
  __int64 v33; // [rsp+68h] [rbp-78h]
  __int64 v34; // [rsp+70h] [rbp-70h]
  _QWORD *v35; // [rsp+80h] [rbp-60h] BYREF
  __int64 v36; // [rsp+88h] [rbp-58h]
  _QWORD v37[10]; // [rsp+90h] [rbp-50h] BYREF

  v26 = a2;
  v28 = 0;
  v29 = 0;
  v30 = 0;
  v35 = v37;
  v37[0] = &v28;
  v27 = a3;
  v3 = 1;
  v36 = 0x400000001LL;
  for ( i = v37; ; i = v35 )
  {
    v5 = (unsigned __int64 *)i[v3 - 1];
    v6 = sub_C934D0(&v26, ",()", 3, 0);
    v32 = 0;
    v7 = v6;
    if ( v27 <= v6 )
      v6 = v27;
    v33 = 0;
    v34 = 0;
    v31.m128i_i64[1] = v6;
    v8 = (__m128i *)v5[1];
    v31.m128i_i64[0] = (__int64)v26;
    if ( v8 == (__m128i *)v5[2] )
    {
      sub_2352B10(v5, v8, &v31);
    }
    else
    {
      if ( v8 )
      {
        *v8 = _mm_loadu_si128(&v31);
        v8[1].m128i_i64[0] = v32;
        v32 = 0;
        v8[1].m128i_i64[1] = v33;
        v33 = 0;
        v8[2].m128i_i64[0] = v34;
        v34 = 0;
        v8 = (__m128i *)v5[1];
      }
      v5[1] = (unsigned __int64)&v8[2].m128i_u64[1];
    }
    sub_234A6B0((unsigned __int64 *)&v32);
    if ( v7 == -1 )
    {
      v18 = v36;
      goto LABEL_16;
    }
    v10 = v27;
    v11 = 0;
    v12 = v26[v7];
    if ( v7 + 1 <= v27 )
    {
      v10 = v7 + 1;
      v11 = v27 - (v7 + 1);
    }
    v13 = &v26[v10];
    v27 = v11;
    v3 = (unsigned int)v36;
    v26 = v13;
    if ( v12 != 44 )
      break;
LABEL_24:
    ;
  }
  if ( v12 == 40 )
  {
    v21 = v5[1] - 24;
    if ( (unsigned __int64)(unsigned int)v36 + 1 > HIDWORD(v36) )
    {
      v24 = v5[1] - 24;
      sub_C8D5F0((__int64)&v35, v37, (unsigned int)v36 + 1LL, 8u, (unsigned int)v36 + 1LL, v9);
      v3 = (unsigned int)v36;
      v21 = v24;
    }
    v35[v3] = v21;
    v3 = (unsigned int)(v36 + 1);
    LODWORD(v36) = v36 + 1;
    goto LABEL_24;
  }
  if ( (_DWORD)v36 == 1 )
    goto LABEL_17;
  v14 = &v13[v11];
  v15 = (__int64)&v13[v11 - 1];
  v16 = &v13[(unsigned int)(v36 - 1)];
  v17 = (_DWORD)v13 + v36 - 1;
  while ( 1 )
  {
    v18 = v17 - (_DWORD)v13;
    LODWORD(v36) = v17 - (_DWORD)v13;
    if ( v13 == v14 )
      break;
    if ( *v13 != 41 )
    {
      if ( !(unsigned __int8)sub_95CB50((const void **)&v26, ",", 1u) )
        goto LABEL_17;
      v3 = (unsigned int)v36;
      goto LABEL_24;
    }
    v20 = v15 - (_QWORD)v13++;
    v26 = v13;
    v27 = v20;
    if ( v16 == v13 )
      goto LABEL_17;
  }
LABEL_16:
  if ( v18 > 1 )
  {
LABEL_17:
    *(_BYTE *)(a1 + 24) = 0;
    goto LABEL_18;
  }
  v22 = v28;
  v28 = 0;
  *(_QWORD *)a1 = v22;
  v23 = v29;
  *(_BYTE *)(a1 + 24) = 1;
  *(_QWORD *)(a1 + 8) = v23;
  v29 = 0;
  *(_QWORD *)(a1 + 16) = v30;
  v30 = 0;
LABEL_18:
  if ( v35 != v37 )
    _libc_free((unsigned __int64)v35);
  sub_234A6B0(&v28);
  return a1;
}
