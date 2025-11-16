// Function: sub_A7C1C0
// Address: 0xa7c1c0
//
__int64 __fastcall sub_A7C1C0(__int64 a1, __int64 a2, char a3, __int64 a4, __int64 a5)
{
  unsigned int v5; // r13d
  __int64 v6; // rax
  unsigned __int64 v7; // rcx
  __int8 *v8; // r12
  size_t v9; // r15
  char *v10; // rax
  __int64 v11; // rsi
  char **v12; // rax
  __int64 v13; // rsi
  char **v14; // r12
  __int64 v15; // rsi
  __int64 v16; // rcx
  _QWORD *v17; // rcx
  char **v18; // r15
  __int64 v19; // rcx
  unsigned __int64 v20; // rdx
  char *v21; // rsi
  __m128i v22; // kr00_16
  __int64 result; // rax
  unsigned __int64 v24; // rax
  char *v25; // rdi
  __m128i v26; // rax
  char **v27; // r12
  __int64 v28; // rsi
  __int64 v29; // rdi
  unsigned __int64 v30; // rax
  unsigned __int64 v31; // r15
  unsigned __int64 v32; // rax
  unsigned __int64 v33; // rcx
  __int64 v34; // rax
  char *v35; // rdx
  unsigned __int64 v36; // rax
  char *v37; // rax
  __m128i v38; // xmm1
  __m128i v44; // [rsp+30h] [rbp-B0h] BYREF
  char *v45; // [rsp+40h] [rbp-A0h] BYREF
  size_t v46; // [rsp+48h] [rbp-98h]
  _QWORD v47[2]; // [rsp+50h] [rbp-90h] BYREF
  __m128i v48; // [rsp+60h] [rbp-80h] BYREF
  __m128i v49; // [rsp+70h] [rbp-70h] BYREF
  char *v50; // [rsp+80h] [rbp-60h] BYREF
  unsigned __int64 v51; // [rsp+88h] [rbp-58h]
  _QWORD v52[10]; // [rsp+90h] [rbp-50h] BYREF

  v5 = 0;
  v50 = "1";
  v51 = 1;
  v52[0] = "1";
  v52[1] = 1;
  v52[2] = "1";
  v52[3] = 1;
  if ( (unsigned __int8)sub_B2D620(a4, a1, a2) )
  {
    v48.m128i_i64[0] = sub_B2D7E0(a4, a1, a2);
    v26.m128i_i64[0] = sub_A72240(v48.m128i_i64);
    v27 = &v50;
    v44 = v26;
    do
    {
      if ( !v44.m128i_i64[1] )
        break;
      LOBYTE(v45) = 44;
      v30 = sub_C931B0(&v44, &v45, 1, 0);
      if ( v30 == -1 )
      {
        v38 = _mm_loadu_si128(&v44);
        v49 = 0u;
        v48 = v38;
      }
      else
      {
        v28 = v30 + 1;
        if ( v30 + 1 > v44.m128i_i64[1] )
        {
          v28 = v44.m128i_i64[1];
          v29 = 0;
        }
        else
        {
          v29 = v44.m128i_i64[1] - v28;
        }
        v48.m128i_i64[0] = v44.m128i_i64[0];
        if ( v30 > v44.m128i_i64[1] )
          v30 = v44.m128i_u64[1];
        v49.m128i_i64[1] = v29;
        v49.m128i_i64[0] = v28 + v44.m128i_i64[0];
        v48.m128i_i64[1] = v30;
      }
      v31 = 0;
      v32 = sub_C935B0(&v48, &unk_3F15413, 6, 0);
      v33 = v48.m128i_u64[1];
      if ( v32 < v48.m128i_i64[1] )
      {
        v31 = v48.m128i_i64[1] - v32;
        v33 = v32;
      }
      v45 = (char *)(v33 + v48.m128i_i64[0]);
      v46 = v31;
      v34 = sub_C93740(&v45, &unk_3F15413, 6, -1);
      v35 = (char *)v46;
      v36 = v34 + 1;
      v44 = _mm_loadu_si128(&v49);
      if ( v36 > v46 )
        v36 = v46;
      *v27 = v45;
      v37 = &v35[v36 - v31];
      if ( v37 > v35 )
        v37 = v35;
      ++v5;
      v27 += 2;
      *(v27 - 1) = v37;
    }
    while ( v5 != 3 );
  }
  v6 = *(_QWORD *)(a5 + 136);
  v7 = *(_QWORD *)(v6 + 24);
  if ( *(_DWORD *)(v6 + 32) > 0x40u )
    v7 = *(_QWORD *)v7;
  if ( !v7 )
  {
    v49.m128i_i8[4] = 48;
    v8 = &v49.m128i_i8[4];
    v45 = (char *)v47;
LABEL_6:
    v9 = 1;
    LOBYTE(v47[0]) = *v8;
    v10 = (char *)v47;
    goto LABEL_7;
  }
  v8 = &v49.m128i_i8[5];
  do
  {
    *--v8 = v7 % 0xA + 48;
    v24 = v7;
    v7 /= 0xAu;
  }
  while ( v24 > 9 );
  v9 = &v49.m128i_u8[5] - (unsigned __int8 *)v8;
  v45 = (char *)v47;
  v44.m128i_i64[0] = &v49.m128i_u8[5] - (unsigned __int8 *)v8;
  if ( (unsigned __int64)(&v49.m128i_u8[5] - (unsigned __int8 *)v8) > 0xF )
  {
    v45 = (char *)sub_22409D0(&v45, &v44, 0);
    v25 = v45;
    v47[0] = v44.m128i_i64[0];
LABEL_27:
    memcpy(v25, v8, v9);
    v9 = v44.m128i_i64[0];
    v10 = v45;
    goto LABEL_7;
  }
  if ( v9 == 1 )
    goto LABEL_6;
  if ( v9 )
  {
    v25 = (char *)v47;
    goto LABEL_27;
  }
  v10 = (char *)v47;
LABEL_7:
  v46 = v9;
  v11 = v5;
  v10[v9] = 0;
  v48 = (__m128i)(unsigned __int64)&v49;
  (&v50)[2 * (unsigned int)(a3 - 120)] = v45;
  if ( a3 - 119 >= v5 )
    v11 = (unsigned int)(a3 - 119);
  v49.m128i_i8[0] = 0;
  v52[2 * (unsigned int)(a3 - 120) - 1] = v46;
  v12 = &v50;
  v13 = 2 * v11;
  v14 = &(&v50)[v13];
  if ( &v12[v13] == v12 )
  {
    v22 = (__m128i)(unsigned __int64)&v49;
  }
  else
  {
    v15 = ((v13 * 8) >> 4) - 1;
    do
    {
      v15 += (__int64)v12[1];
      v12 += 2;
    }
    while ( v14 != v12 );
    sub_2240E30(&v48, v15);
    if ( v51 > 0x3FFFFFFFFFFFFFFFLL - v48.m128i_i64[1] )
      goto LABEL_49;
    sub_2241490(&v48, v50, v51, v16);
    v17 = v52;
    v18 = (char **)v52;
    if ( v14 != v52 )
    {
      while ( v48.m128i_i64[1] != 0x3FFFFFFFFFFFFFFFLL )
      {
        sub_2241490(&v48, ",", 1, v17);
        v20 = (unsigned __int64)v18[1];
        v21 = *v18;
        if ( v20 > 0x3FFFFFFFFFFFFFFFLL - v48.m128i_i64[1] )
          break;
        v18 += 2;
        sub_2241490(&v48, v21, v20, v19);
        if ( v14 == v18 )
          goto LABEL_17;
      }
LABEL_49:
      sub_4262D8((__int64)"basic_string::append");
    }
LABEL_17:
    v22 = v48;
  }
  result = sub_B2CD60(a4, a1, a2, v22.m128i_i64[0], v22.m128i_i64[1]);
  if ( (__m128i *)v48.m128i_i64[0] != &v49 )
    result = j_j___libc_free_0(v48.m128i_i64[0], v49.m128i_i64[0] + 1);
  if ( v45 != (char *)v47 )
    return j_j___libc_free_0(v45, v47[0] + 1LL);
  return result;
}
