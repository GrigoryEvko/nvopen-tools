// Function: sub_7AFFB0
// Address: 0x7affb0
//
__int64 __fastcall sub_7AFFB0(
        char *a1,
        int a2,
        __int64 a3,
        __int64 **a4,
        int a5,
        int a6,
        char **a7,
        __int64 *a8,
        _DWORD *a9,
        int *a10,
        _DWORD *a11,
        _QWORD *a12,
        int a13)
{
  char *v16; // rax
  bool v17; // r13
  char *v18; // r13
  _BOOL4 v19; // r8d
  __int64 v20; // rax
  __int64 v22; // rbx
  char *v23; // rax
  __m128i *v24; // rax
  __int64 v25; // rax
  __int64 **v26; // r12
  __int64 v27; // r14
  char *v28; // rdi
  __int64 v29; // rax
  _QWORD *v30; // rax
  __int64 v31; // rdi
  __int64 v32; // rax
  __m128i *v33; // rax
  __m128i *v34; // r14
  size_t v35; // rax
  char *v36; // rax
  size_t v37; // rax
  char *v38; // rax
  size_t v39; // rax
  char *v40; // rax
  char *v41; // rax
  char *v42; // rdx
  bool v44; // [rsp+13h] [rbp-7Dh]
  char *s; // [rsp+18h] [rbp-78h]
  __int64 v47; // [rsp+20h] [rbp-70h]
  _BOOL4 v48; // [rsp+28h] [rbp-68h]
  __m128i *v49; // [rsp+30h] [rbp-60h]
  char *v50; // [rsp+38h] [rbp-58h]
  __m128i v51; // [rsp+40h] [rbp-50h] BYREF
  __m128i v52[4]; // [rsp+50h] [rbp-40h] BYREF

  s = a1;
  *a12 = 0;
  *a8 = 0;
  *a9 = 0;
  *a11 = 0;
  v16 = sub_7222C0(a1);
  v44 = 1;
  if ( !a5 )
    v44 = *v16 == 0;
  v17 = a6 != 0;
  if ( !a2 || v17 )
  {
    v20 = sub_685DD0((__int64)a1, 15, a10, (__int64)a11);
    *a8 = v20;
    v19 = v20 != 0;
    if ( !v17 || a2 == 0 || v20 )
    {
LABEL_7:
      if ( v20 )
        goto LABEL_8;
      goto LABEL_37;
    }
    v18 = a1;
  }
  else
  {
    v18 = (char *)byte_3F871B3;
    v19 = sub_7215C0((unsigned __int8 *)a1);
    if ( v19 )
    {
      v20 = sub_685DD0((__int64)a1, 15, a10, (__int64)a11);
      *a8 = v20;
      goto LABEL_7;
    }
  }
  if ( !a3 )
  {
    if ( !a13 )
      sub_685220(0xB6u, (__int64)a1);
    if ( v19 )
      goto LABEL_58;
LABEL_37:
    *a7 = 0;
    return 0;
  }
  v49 = 0;
  v22 = 0;
  v23 = 0;
  v48 = v19;
  while ( 1 )
  {
    v50 = *(char **)a3;
    if ( *(char **)a3 == v23 )
      goto LABEL_35;
    v49 = 0;
    if ( a5 )
      goto LABEL_18;
    v51.m128i_i64[1] = *(_QWORD *)a3;
    v51.m128i_i64[0] = (__int64)qword_4F076B0;
    v52[0] = (__m128i)(unsigned __int64)s;
    v27 = sub_881B20(qword_4F08500, &v51, 1);
    v49 = *(__m128i **)v27;
    if ( !*(_QWORD *)v27 )
      break;
    v28 = *(char **)(*(_QWORD *)v27 + 24LL);
    if ( v28 )
      goto LABEL_40;
LABEL_34:
    v23 = v50;
LABEL_35:
    a3 = *(_QWORD *)(a3 + 16);
    if ( !a3 )
    {
      if ( v48 )
        goto LABEL_20;
      goto LABEL_37;
    }
  }
  v33 = (__m128i *)sub_822B10(32);
  v33->m128i_i64[0] = 0;
  v33->m128i_i64[1] = 0;
  v33[1].m128i_i64[0] = 0;
  v33[1].m128i_i64[1] = 0;
  *(_QWORD *)v27 = v33;
  v34 = v33;
  v49 = v33;
  *v33 = _mm_loadu_si128(&v51);
  v33[1] = _mm_loadu_si128(v52);
  v35 = strlen(s);
  v36 = (char *)sub_822B10(v35 + 1);
  v34[1].m128i_i64[0] = (__int64)v36;
  strcpy(v36, s);
  v28 = (char *)v34[1].m128i_i64[1];
  if ( !v28 )
  {
LABEL_18:
    if ( v48 )
      goto LABEL_19;
    goto LABEL_43;
  }
LABEL_40:
  *a9 = 0;
  v51.m128i_i64[0] = 0;
  *a8 = 0;
  *a11 = 0;
  if ( sub_7AFEF0(v28, v51.m128i_i64, 0, 0) )
  {
    v18 = v28;
    *a9 = 1;
    goto LABEL_19;
  }
  v29 = sub_685DD0((__int64)v28, 15, a10, (__int64)a11);
  *a8 = v29;
  if ( v29 )
  {
    v18 = v28;
    goto LABEL_19;
  }
LABEL_43:
  v30 = sub_720CF0(v50, s, 0);
  v18 = (char *)v30[4];
  v22 = (__int64)v30;
  if ( !v44 )
  {
    v51.m128i_i64[0] = 0;
    *a9 = 0;
    *a8 = 0;
    *a11 = 0;
    v48 = sub_7AFEF0(v18, v51.m128i_i64, 0, 0);
    if ( v48 )
    {
LABEL_45:
      *a9 = 1;
      goto LABEL_19;
    }
    v32 = sub_685DD0((__int64)v18, 15, a10, (__int64)a11);
    *a8 = v32;
    if ( v32 )
      goto LABEL_19;
    v23 = v50;
    goto LABEL_35;
  }
  v31 = qword_4F17F60;
  if ( !qword_4F17F60 )
  {
    qword_4F17F60 = sub_8237A0(128);
    v31 = qword_4F17F60;
  }
  sub_823800(v31);
  sub_8238B0(qword_4F17F60, *(_QWORD *)(v22 + 32), *(_QWORD *)(v22 + 16));
  if ( !a4 )
  {
    v48 = 0;
    v23 = v50;
    goto LABEL_35;
  }
  v47 = a3;
  v26 = a4;
  while ( 1 )
  {
    sub_722380((const char *)v26[1], v22);
    v18 = *(char **)(v22 + 32);
    v51.m128i_i64[0] = 0;
    *a9 = 0;
    *a8 = 0;
    *a11 = 0;
    if ( sub_7AFEF0(v18, v51.m128i_i64, 0, 0) )
    {
      a3 = v47;
      goto LABEL_45;
    }
    v25 = sub_685DD0((__int64)v18, 15, a10, (__int64)a11);
    *a8 = v25;
    if ( v25 )
      break;
    if ( *v26 )
    {
      sub_823800(v22);
      sub_8238B0(v22, *(_QWORD *)(qword_4F17F60 + 32), *(_QWORD *)(qword_4F17F60 + 16));
      v26 = (__int64 **)*v26;
      if ( v26 )
        continue;
    }
    v48 = 0;
    a3 = v47;
    goto LABEL_34;
  }
  a3 = v47;
LABEL_19:
  *a12 = a3;
LABEL_20:
  if ( v22 )
  {
    if ( *(char **)(v22 + 32) == v18 )
    {
      v41 = (char *)sub_7279A0(*(_QWORD *)(v22 + 16));
      v42 = strcpy(v41, v18);
      *a7 = v42;
      if ( v49 )
      {
        v18 = v42;
        goto LABEL_27;
      }
    }
    else
    {
      if ( !v49 )
        goto LABEL_58;
      v24 = v49;
LABEL_24:
      if ( (char *)v24[1].m128i_i64[1] == v18 )
      {
        v37 = strlen(v18);
        v38 = (char *)sub_7279A0(v37 + 1);
        v18 = strcpy(v38, v18);
      }
      *a7 = v18;
LABEL_27:
      if ( !v49[1].m128i_i64[1] )
      {
        v39 = strlen(v18);
        v40 = (char *)sub_822B10(v39 + 1);
        v49[1].m128i_i64[1] = (__int64)v40;
        strcpy(v40, v18);
      }
    }
    return 1;
  }
  v24 = v49;
  if ( v49 )
    goto LABEL_24;
LABEL_58:
  s = v18;
LABEL_8:
  *a7 = s;
  return 1;
}
