// Function: sub_373ABD0
// Address: 0x373abd0
//
unsigned __int64 __fastcall sub_373ABD0(__int64 *a1, __int64 a2, char a3)
{
  __int64 *v5; // r12
  int v7; // r13d
  unsigned __int64 v8; // rax
  __int16 v9; // r13
  unsigned __int8 *v10; // rsi
  _QWORD *v11; // r13
  __int64 v12; // rdi
  __int64 v13; // rax
  unsigned __int64 result; // rax
  __int64 v15; // rdx
  _QWORD *v16; // rax
  _QWORD *v17; // rcx
  __int64 (*v18)(); // rcx
  unsigned __int8 v19; // al
  __int64 v20; // rdx
  __int64 v21; // rdi
  __int64 v22; // rax
  int v23; // r13d
  __int64 v24; // rax
  unsigned __int8 v25; // dl
  _BYTE **v26; // rax
  __int64 v27; // rax
  __int64 v28; // rax
  unsigned __int64 **v29; // r12
  const char *v30; // rax
  unsigned __int64 v31; // rdx
  const char *v32; // r8
  size_t v33; // r9
  __m128i *v34; // rax
  __int64 v35; // rdi
  __int64 v36; // rax
  const char *v37; // rax
  char *v38; // rdi
  size_t n; // [rsp+0h] [rbp-A0h]
  __int64 (*src)(); // [rsp+8h] [rbp-98h]
  const char *srca; // [rsp+8h] [rbp-98h]
  unsigned __int64 v42; // [rsp+18h] [rbp-88h] BYREF
  unsigned __int64 v43[2]; // [rsp+20h] [rbp-80h] BYREF
  __m128i v44; // [rsp+30h] [rbp-70h] BYREF
  const char *v45; // [rsp+40h] [rbp-60h] BYREF
  unsigned __int64 *v46; // [rsp+48h] [rbp-58h]
  _QWORD v47[2]; // [rsp+50h] [rbp-50h] BYREF
  __int16 v48; // [rsp+60h] [rbp-40h]

  v5 = a1 + 11;
  v7 = -(*(_WORD *)(*(_QWORD *)(a2 + 8) + 20LL) == 0);
  v8 = sub_A777F0(0x30u, a1 + 11);
  v9 = (v7 & 0x2F) + 5;
  if ( v8 )
  {
    *(_QWORD *)(v8 + 8) = 0;
    *(_QWORD *)v8 = v8 | 4;
    *(_QWORD *)(v8 + 16) = 0;
    *(_DWORD *)(v8 + 24) = -1;
    *(_WORD *)(v8 + 28) = v9;
    *(_BYTE *)(v8 + 30) = 0;
    *(_QWORD *)(v8 + 32) = 0;
    *(_QWORD *)(v8 + 40) = 0;
  }
  v10 = *(unsigned __int8 **)(a2 + 8);
  v42 = v8;
  sub_324C3F0((__int64)a1, v10, v8);
  *(_QWORD *)(a2 + 24) = v42;
  v11 = (_QWORD *)a1[23];
  if ( (unsigned int)(*(_DWORD *)(v11[25] + 544LL) - 42) > 1 )
    goto LABEL_5;
  v12 = a1[26];
  if ( *(_DWORD *)(v12 + 6224) != 1 )
    goto LABEL_5;
  v15 = *(_QWORD *)(a2 + 8);
  if ( !*(_WORD *)(v15 + 20) )
    goto LABEL_5;
  if ( *(_BYTE *)(v12 + 3724) )
  {
    v16 = *(_QWORD **)(v12 + 3704);
    v17 = &v16[*(unsigned int *)(v12 + 3716)];
    if ( v16 == v17 )
      goto LABEL_5;
    while ( a2 != *v16 )
    {
      if ( v17 == ++v16 )
        goto LABEL_5;
    }
  }
  else
  {
    if ( !sub_C8CA60(v12 + 3696, a2) )
    {
LABEL_5:
      if ( a3 )
      {
        sub_3736380(a1, a2, v42);
      }
      else
      {
        v45 = (const char *)a2;
        v46 = &v42;
        v13 = *(char *)(a2 + 88);
        v47[0] = a1;
        if ( (_BYTE)v13 == 0xFF )
          abort();
        funcs_373ACCA[v13]();
      }
      return v42;
    }
    v11 = (_QWORD *)a1[23];
    v15 = *(_QWORD *)(a2 + 8);
  }
  v18 = *(__int64 (**)())(*v11 + 464LL);
  v19 = *(_BYTE *)(v15 - 16);
  if ( (v19 & 2) != 0 )
    v20 = *(_QWORD *)(v15 - 32);
  else
    v20 = v15 - 16 - 8LL * ((v19 >> 2) & 0xF);
  v21 = *(_QWORD *)(v20 + 8);
  if ( v21 )
  {
    src = *(__int64 (**)())(*v11 + 464LL);
    v22 = sub_B91420(v21);
    v18 = src;
    v21 = v22;
  }
  if ( v18 == sub_31D4890 )
    goto LABEL_5;
  v23 = ((__int64 (__fastcall *)(_QWORD *, __int64))v18)(v11, v21);
  if ( v23 < 0 )
    goto LABEL_5;
  v24 = *(_QWORD *)(a2 + 8);
  v25 = *(_BYTE *)(v24 - 16);
  v26 = (v25 & 2) != 0 ? *(_BYTE ***)(v24 - 32) : (_BYTE **)(v24 - 16 - 8LL * ((v25 >> 2) & 0xF));
  v27 = sub_AE7A60(*v26);
  if ( !sub_AF3E00(v27, **(_QWORD **)(a1[23] + 232)) )
    goto LABEL_5;
  v28 = sub_A777F0(0x10u, v5);
  v29 = (unsigned __int64 **)v28;
  if ( v28 )
  {
    *(_QWORD *)v28 = 0;
    *(_DWORD *)(v28 + 8) = 0;
  }
  LODWORD(v45) = 65547;
  sub_3249A20(a1, (unsigned __int64 **)v28, 0, 65547, 3);
  v30 = sub_BD5D20(**(_QWORD **)(a1[23] + 232));
  v32 = v30;
  v33 = v31;
  if ( v30 )
  {
    v43[0] = v31;
    v45 = (const char *)v47;
    if ( v31 > 0xF )
    {
      n = v31;
      srca = v30;
      v37 = (const char *)sub_22409D0((__int64)&v45, v43, 0);
      v32 = srca;
      v33 = n;
      v45 = v37;
      v38 = (char *)v37;
      v47[0] = v43[0];
    }
    else
    {
      if ( v31 == 1 )
      {
        LOBYTE(v47[0]) = *v30;
        goto LABEL_35;
      }
      if ( !v31 )
      {
LABEL_35:
        v46 = (unsigned __int64 *)v43[0];
        v45[v43[0]] = 0;
        if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - (_QWORD)v46) <= 6 )
          sub_4262D8((__int64)"basic_string::append");
        goto LABEL_36;
      }
      v38 = (char *)v47;
    }
    memcpy(v38, v32, v33);
    goto LABEL_35;
  }
  v46 = 0;
  v45 = (const char *)v47;
  LOBYTE(v47[0]) = 0;
LABEL_36:
  v34 = (__m128i *)sub_2241490((unsigned __int64 *)&v45, "_param_", 7u);
  v43[0] = (unsigned __int64)&v44;
  if ( (__m128i *)v34->m128i_i64[0] == &v34[1] )
  {
    v44 = _mm_loadu_si128(v34 + 1);
  }
  else
  {
    v43[0] = v34->m128i_i64[0];
    v44.m128i_i64[0] = v34[1].m128i_i64[0];
  }
  v43[1] = v34->m128i_u64[1];
  v34->m128i_i64[0] = (__int64)v34[1].m128i_i64;
  v34->m128i_i64[1] = 0;
  v34[1].m128i_i8[0] = 0;
  if ( v45 != (const char *)v47 )
    j_j___libc_free_0((unsigned __int64)v45);
  v35 = *(_QWORD *)(a1[23] + 216);
  LODWORD(v47[0]) = v23;
  v45 = (const char *)v43;
  v48 = 2564;
  v36 = sub_E6C460(v35, &v45);
  sub_3249320(a1, v29, 0, 1, v36);
  sub_32498C0(a1, v42, 2, (__int64)v29);
  LODWORD(v45) = 65547;
  sub_3249A20(a1, (unsigned __int64 **)(v42 + 8), 51, 65547, 7);
  result = v42;
  if ( (__m128i *)v43[0] != &v44 )
  {
    j_j___libc_free_0(v43[0]);
    return v42;
  }
  return result;
}
