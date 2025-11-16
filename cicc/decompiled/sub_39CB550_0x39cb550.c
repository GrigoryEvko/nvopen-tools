// Function: sub_39CB550
// Address: 0x39cb550
//
void __fastcall sub_39CB550(__int64 *a1, __m128i *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rbx
  unsigned __int64 v7; // r13
  __int64 v8; // rax
  __int64 v9; // r10
  __int64 v10; // r15
  _BYTE *v11; // r14
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rsi
  unsigned __int64 v15; // rdi
  unsigned __int64 *v16; // rdi
  __m128i *v17; // rsi
  char v18; // al
  __int64 v19; // rax
  unsigned __int64 v20; // rdi
  __int64 v21; // rax
  int v22; // eax
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // r14
  __int64 v27; // rdx
  __int64 v28; // rsi
  __int64 v29; // rdx
  __int64 v30; // rdi
  __int64 v31; // rdx
  __int64 v32; // rax
  const void *v33; // r14
  __int64 v34; // rax
  size_t v35; // rdx
  size_t v36; // r15
  const void *v37; // rdi
  __int64 v38; // rdx
  __int64 v39; // r12
  __int64 v40; // rdx
  __int64 v41; // rax
  __int64 v42; // rdx
  __int64 v43; // rax
  _BYTE *v44; // rdx
  __int64 v45; // rax
  size_t v46; // rdx
  size_t v47; // rcx
  unsigned int v48; // eax
  __int64 v49; // r8
  __int64 v51; // [rsp+10h] [rbp-70h]
  __int64 v52; // [rsp+18h] [rbp-68h]
  __int64 v53; // [rsp+18h] [rbp-68h]
  __int64 v54; // [rsp+18h] [rbp-68h]
  __int64 v55; // [rsp+18h] [rbp-68h]
  __int64 v56; // [rsp+18h] [rbp-68h]
  __int64 v57; // [rsp+18h] [rbp-68h]
  unsigned __int32 v58; // [rsp+20h] [rbp-60h]
  char v59; // [rsp+25h] [rbp-5Bh]
  char v60; // [rsp+26h] [rbp-5Ah]
  bool v61; // [rsp+27h] [rbp-59h]
  __int64 v62; // [rsp+30h] [rbp-50h]
  char v63; // [rsp+30h] [rbp-50h]
  __int64 *v64; // [rsp+38h] [rbp-48h]
  __m128i v65; // [rsp+40h] [rbp-40h] BYREF

  v51 = (__int64)a2;
  v62 = a4 + 16 * a5;
  if ( a4 == v62 )
  {
    if ( (unsigned int)(*(_DWORD *)(*(_QWORD *)(a1[24] + 232) + 504LL) - 34) > 1 )
    {
      if ( !*(_BYTE *)(a1[25] + 4498) )
        return;
      v60 = 0;
      v7 = 0;
LABEL_80:
      v43 = 5LL - *(unsigned int *)(a3 + 8);
      v44 = *(_BYTE **)(a3 + 8 * v43);
      if ( v44 )
      {
        v45 = sub_161E970(*(_QWORD *)(a3 + 8 * v43));
        v47 = v46;
        v44 = (_BYTE *)v45;
      }
      else
      {
        v47 = 0;
      }
      sub_39A40D0(a1, v51, v44, v47);
      goto LABEL_52;
    }
    v19 = a1[25];
    if ( *(_DWORD *)(v19 + 6584) != 1 )
    {
      v60 = 0;
      v7 = 0;
      goto LABEL_51;
    }
    v60 = 0;
    v7 = 0;
    v49 = 5;
    v64 = 0;
LABEL_86:
    v65.m128i_i32[0] = 65547;
    sub_39A3560((__int64)a1, (__int64 *)(v51 + 8), 51, (__int64)&v65, v49);
    goto LABEL_48;
  }
  v59 = 0;
  v6 = a4;
  v61 = a5 == 1;
  v7 = 0;
  v64 = 0;
  v60 = 0;
  while ( 1 )
  {
    v10 = *(_QWORD *)(v6 + 8);
    v11 = *(_BYTE **)v6;
    if ( v10 )
    {
      if ( v61 )
        break;
    }
    if ( !v11 )
    {
      if ( !v10 )
        goto LABEL_19;
LABEL_24:
      if ( (unsigned __int8)sub_15B1550(v10, (__int64)a2, a3) )
      {
        if ( !v64 )
          goto LABEL_26;
        goto LABEL_5;
      }
      goto LABEL_19;
    }
LABEL_3:
    if ( (v11[33] & 3) != 1 )
    {
      if ( !v64 )
      {
LABEL_26:
        v12 = sub_145CDC0(0x10u, a1 + 11);
        v64 = (__int64 *)v12;
        if ( v12 )
        {
          *(_QWORD *)v12 = 0;
          *(_DWORD *)(v12 + 8) = 0;
        }
        v53 = a1[24];
        v13 = sub_22077B0(0x70u);
        if ( v13 )
        {
          v14 = v53;
          v54 = v13;
          sub_39A1E10(v13, v14, (__int64)a1, (__int64)v64);
          v13 = v54;
        }
        if ( v7 )
        {
          v15 = *(_QWORD *)(v7 + 8);
          if ( v15 != v7 + 24 )
          {
            v55 = v13;
            _libc_free(v15);
            v13 = v55;
          }
          v56 = v13;
          j_j___libc_free_0(v7);
          v13 = v56;
        }
        v60 = 1;
        v7 = v13;
      }
LABEL_5:
      if ( v10 )
      {
        if ( (unsigned int)(*(_DWORD *)(*(_QWORD *)(a1[24] + 232) + 504LL) - 34) <= 1 && *(_DWORD *)(a1[25] + 6584) == 1 )
        {
          v25 = sub_15C4660((_QWORD *)v10, &v65);
          if ( v10 != v25 )
          {
            v10 = v25;
            v58 = v65.m128i_i32[0];
            if ( !v59 )
              v59 = 1;
          }
        }
        sub_399FD50(v7, v10);
      }
      if ( v11 )
      {
        v8 = sub_396EAF0(a1[24], (__int64)v11);
        v9 = v8;
        if ( (v11[33] & 0x1C) != 0 )
        {
          v52 = v8;
          if ( !(unsigned __int8)sub_17006E0(*(_QWORD *)(a1[24] + 232)) )
          {
            v21 = sub_396DDB0(a1[24]);
            v22 = sub_15A9520(v21, 0);
            if ( *(_BYTE *)(a1[25] + 4513) )
            {
              sub_39A35E0((__int64)a1, v64, 11, 252);
              v48 = sub_39BFF80(a1[25] + 5512, v52, 1);
              sub_39A35E0((__int64)a1, v64, 15, v48);
            }
            else
            {
              sub_39A35E0((__int64)a1, v64, 11, 2LL * (v22 != 4) + 12);
              v23 = sub_396DD80(a1[24]);
              v24 = (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v23 + 112LL))(v23, v52);
              sub_39C9B70((__int64)a1, (unsigned __int64 **)v64, 15, v24);
            }
            sub_39A35E0((__int64)a1, v64, 11, (-(__int64)(*(_BYTE *)(a1[25] + 4496) == 0) & 0xFFFFFFFFFFFFFFBBLL) + 224);
          }
        }
        else
        {
          v16 = (unsigned __int64 *)a1[25];
          v65.m128i_i64[0] = v8;
          v65.m128i_i64[1] = (__int64)a1;
          v17 = (__m128i *)v16[77];
          if ( v17 == (__m128i *)v16[78] )
          {
            v57 = v8;
            sub_39CAC70(v16 + 76, v17, &v65);
            v9 = v57;
          }
          else
          {
            if ( v17 )
            {
              *v17 = _mm_loadu_si128(&v65);
              v17 = (__m128i *)v16[77];
            }
            v16[77] = (unsigned __int64)&v17[1];
          }
          sub_39A39D0((__int64)a1, v64, v9);
        }
        if ( (unsigned int)(*(_DWORD *)(*(_QWORD *)(a1[24] + 232) + 504LL) - 34) <= 1
          && *(_DWORD *)(a1[25] + 6584) == 1
          && !v59 )
        {
          v59 = 1;
          v58 = sub_3989E90(*(_DWORD *)(*(_QWORD *)v11 + 8LL) >> 8);
        }
      }
      if ( !*(_DWORD *)(v7 + 76) )
        *(_DWORD *)(v7 + 76) = 2;
      v65 = 0u;
      if ( v10 )
        v65 = *(__m128i *)(v10 + 24);
      a2 = &v65;
      sub_399FAC0(v7, &v65, 0);
    }
LABEL_19:
    v6 += 16;
    if ( v62 == v6 )
      goto LABEL_46;
  }
  v18 = sub_15B1550(*(_QWORD *)(v6 + 8), (__int64)a2, a3);
  if ( !v18 )
  {
    if ( !v11 )
      goto LABEL_24;
    goto LABEL_3;
  }
  v63 = v18;
  sub_39A37F0((__int64)a1, v51, 1, *(_QWORD *)(*(_QWORD *)(v10 + 24) + 8LL));
  v60 = v63;
LABEL_46:
  if ( (unsigned int)(*(_DWORD *)(*(_QWORD *)(a1[24] + 232) + 504LL) - 34) <= 1 && *(_DWORD *)(a1[25] + 6584) == 1 )
  {
    v49 = v58;
    if ( !v59 )
      v49 = 5;
    goto LABEL_86;
  }
LABEL_48:
  if ( v64 )
  {
    sub_399FD30(v7);
    sub_39A4520(a1, v51, 2, *(__int64 ***)(v7 + 104));
  }
  v19 = a1[25];
LABEL_51:
  if ( *(_BYTE *)(v19 + 4498) )
    goto LABEL_80;
LABEL_52:
  if ( v60 )
  {
    v26 = a1[25];
    v27 = *(unsigned int *)(a3 + 8);
    v28 = *(_QWORD *)(a3 + 8 * (1 - v27));
    if ( v28 )
      v28 = sub_161E970(*(_QWORD *)(a3 + 8 * (1 - v27)));
    else
      v29 = 0;
    sub_398FC80(v26, v28, v29, v51);
    v30 = *(_QWORD *)(a3 + 8 * (5LL - *(unsigned int *)(a3 + 8)));
    if ( v30 )
    {
      sub_161E970(v30);
      if ( v31 )
      {
        v32 = *(unsigned int *)(a3 + 8);
        v33 = *(const void **)(a3 + 8 * (5 - v32));
        if ( v33 )
        {
          v34 = sub_161E970((__int64)v33);
          v36 = v35;
          v33 = (const void *)v34;
          v37 = *(const void **)(a3 + 8 * (1LL - *(unsigned int *)(a3 + 8)));
          if ( v37 )
            goto LABEL_71;
          v38 = 0;
          goto LABEL_72;
        }
        v37 = *(const void **)(a3 + 8 * (1 - v32));
        if ( v37 )
        {
          v36 = 0;
LABEL_71:
          v37 = (const void *)sub_161E970((__int64)v37);
LABEL_72:
          if ( v36 != v38 || v36 && memcmp(v37, v33, v36) )
          {
            v39 = a1[25];
            if ( *(_BYTE *)(v39 + 4498) )
            {
              v40 = *(unsigned int *)(a3 + 8);
              v41 = *(_QWORD *)(a3 + 8 * (5 - v40));
              if ( v41 )
                v41 = sub_161E970(*(_QWORD *)(a3 + 8 * (5 - v40)));
              else
                v42 = 0;
              sub_398FC80(v39, v41, v42, v51);
            }
          }
        }
      }
    }
  }
  if ( v7 )
  {
    v20 = *(_QWORD *)(v7 + 8);
    if ( v20 != v7 + 24 )
      _libc_free(v20);
    j_j___libc_free_0(v7);
  }
}
