// Function: sub_2674850
// Address: 0x2674850
//
__m128i *__fastcall sub_2674850(__m128i *a1, __int64 a2)
{
  unsigned int v3; // ebx
  unsigned __int64 v4; // rsi
  __m128i *v5; // rax
  __int64 v6; // rcx
  __m128i *v7; // rax
  size_t v8; // rcx
  __m128i *v9; // r9
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // rdi
  __m128i *v12; // rax
  unsigned __int64 v13; // rcx
  __m128i *v14; // rdx
  __m128i *v15; // rax
  size_t v16; // rcx
  __m128i *v17; // r9
  unsigned __int64 v18; // rax
  unsigned __int64 v19; // rdi
  __m128i *v20; // rax
  unsigned __int64 v21; // rsi
  __m128i *v22; // rdx
  __m128i *v23; // rax
  __int64 v24; // rsi
  _OWORD *v25; // rdi
  _QWORD *v28; // rsi
  _QWORD *v29; // rcx
  __int64 v30; // rdx
  _QWORD *v31; // rax
  unsigned int v32; // r14d
  unsigned int v33; // r13d
  unsigned __int64 v34; // rdx
  unsigned int v35; // ecx
  unsigned __int64 v36; // rsi
  unsigned int v37; // r9d
  unsigned __int64 v38; // rsi
  unsigned __int64 v39; // rdx
  unsigned int v40; // ecx
  unsigned __int64 v41; // rsi
  unsigned int v42; // r9d
  unsigned __int64 v43; // rsi
  unsigned __int64 v44; // rdx
  unsigned int v45; // ecx
  unsigned __int64 v46; // rsi
  unsigned int v47; // r8d
  _BYTE *v48; // [rsp+20h] [rbp-130h] BYREF
  int v49; // [rsp+28h] [rbp-128h]
  _QWORD v50[2]; // [rsp+30h] [rbp-120h] BYREF
  __m128i *v51; // [rsp+40h] [rbp-110h] BYREF
  __int64 v52; // [rsp+48h] [rbp-108h]
  __m128i v53; // [rsp+50h] [rbp-100h] BYREF
  __m128i *v54; // [rsp+60h] [rbp-F0h] BYREF
  size_t v55; // [rsp+68h] [rbp-E8h]
  __m128i v56; // [rsp+70h] [rbp-E0h] BYREF
  char *v57; // [rsp+80h] [rbp-D0h] BYREF
  size_t v58; // [rsp+88h] [rbp-C8h]
  _QWORD v59[2]; // [rsp+90h] [rbp-C0h] BYREF
  __m128i *v60; // [rsp+A0h] [rbp-B0h] BYREF
  __int64 v61; // [rsp+A8h] [rbp-A8h]
  __m128i v62; // [rsp+B0h] [rbp-A0h] BYREF
  __m128i *v63; // [rsp+C0h] [rbp-90h] BYREF
  size_t v64; // [rsp+C8h] [rbp-88h]
  __m128i v65; // [rsp+D0h] [rbp-80h] BYREF
  char *v66; // [rsp+E0h] [rbp-70h] BYREF
  size_t v67; // [rsp+E8h] [rbp-68h]
  _QWORD v68[2]; // [rsp+F0h] [rbp-60h] BYREF
  _OWORD *v69; // [rsp+100h] [rbp-50h] BYREF
  __int64 v70; // [rsp+108h] [rbp-48h]
  _OWORD v71[4]; // [rsp+110h] [rbp-40h] BYREF

  if ( !*(_DWORD *)(a2 + 240) )
    goto LABEL_2;
  v28 = *(_QWORD **)(a2 + 232);
  v29 = &v28[16 * (unsigned __int64)*(unsigned int *)(a2 + 248)];
  if ( v28 == v29 )
    goto LABEL_2;
  while ( 1 )
  {
    v30 = *v28;
    v31 = v28;
    if ( *v28 != -4096 && v30 != -8192 )
      break;
    v28 += 16;
    if ( v29 == v28 )
      goto LABEL_2;
  }
  if ( v29 == v28 )
  {
LABEL_2:
    v3 = 0;
    v66 = (char *)v68;
    sub_2240A50((__int64 *)&v66, 1u, 0);
    sub_2554A60(v66, v67, 0);
    v57 = (char *)v59;
    sub_2240A50((__int64 *)&v57, 1u, 0);
    sub_2554A60(v57, v58, 0);
    v4 = 1;
  }
  else
  {
    v32 = 0;
    v3 = 0;
    v33 = 0;
    do
    {
      if ( v30 )
      {
        ++v33;
        v3 += *((unsigned __int8 *)v31 + 8);
        if ( *((_BYTE *)v31 + 9) )
          v32 += *((unsigned __int8 *)v31 + 10);
      }
      v31 += 16;
      if ( v31 == v29 )
        break;
      while ( 1 )
      {
        v30 = *v31;
        if ( *v31 != -8192 && v30 != -4096 )
          break;
        v31 += 16;
        if ( v29 == v31 )
          goto LABEL_65;
      }
    }
    while ( v29 != v31 );
LABEL_65:
    if ( v33 <= 9 )
    {
      v38 = 1;
    }
    else if ( v33 <= 0x63 )
    {
      v38 = 2;
    }
    else if ( v33 <= 0x3E7 )
    {
      v38 = 3;
    }
    else
    {
      v34 = v33;
      if ( v33 <= 0x270F )
      {
        v38 = 4;
      }
      else
      {
        v35 = 1;
        do
        {
          v36 = v34;
          v37 = v35;
          v35 += 4;
          v34 /= 0x2710u;
          if ( v36 <= 0x1869F )
          {
            v38 = v35;
            goto LABEL_75;
          }
          if ( (unsigned int)v34 <= 0x63 )
          {
            v38 = v37 + 5;
            goto LABEL_75;
          }
          if ( (unsigned int)v34 <= 0x3E7 )
          {
            v38 = v37 + 6;
            goto LABEL_75;
          }
        }
        while ( (unsigned int)v34 > 0x270F );
        v38 = v37 + 7;
      }
    }
LABEL_75:
    v66 = (char *)v68;
    sub_2240A50((__int64 *)&v66, v38, 0);
    sub_2554A60(v66, v67, v33);
    if ( v32 <= 9 )
    {
      v43 = 1;
    }
    else if ( v32 <= 0x63 )
    {
      v43 = 2;
    }
    else if ( v32 <= 0x3E7 )
    {
      v43 = 3;
    }
    else
    {
      v39 = v32;
      if ( v32 <= 0x270F )
      {
        v43 = 4;
      }
      else
      {
        v40 = 1;
        do
        {
          v41 = v39;
          v42 = v40;
          v40 += 4;
          v39 /= 0x2710u;
          if ( v41 <= 0x1869F )
          {
            v43 = v40;
            goto LABEL_85;
          }
          if ( (unsigned int)v39 <= 0x63 )
          {
            v43 = v42 + 5;
            goto LABEL_85;
          }
          if ( (unsigned int)v39 <= 0x3E7 )
          {
            v43 = v42 + 6;
            goto LABEL_85;
          }
        }
        while ( (unsigned int)v39 > 0x270F );
        v43 = v42 + 7;
      }
    }
LABEL_85:
    v57 = (char *)v59;
    sub_2240A50((__int64 *)&v57, v43, 0);
    sub_2554A60(v57, v58, v32);
    if ( v3 <= 9 )
    {
      v4 = 1;
    }
    else if ( v3 <= 0x63 )
    {
      v4 = 2;
    }
    else if ( v3 <= 0x3E7 )
    {
      v4 = 3;
    }
    else
    {
      v44 = v3;
      if ( v3 <= 0x270F )
      {
        v4 = 4;
      }
      else
      {
        v45 = 1;
        do
        {
          v46 = v44;
          v47 = v45;
          v45 += 4;
          v44 /= 0x2710u;
          if ( v46 <= 0x1869F )
          {
            v4 = v45;
            goto LABEL_3;
          }
          if ( (unsigned int)v44 <= 0x63 )
          {
            v4 = v47 + 5;
            goto LABEL_3;
          }
          if ( (unsigned int)v44 <= 0x3E7 )
          {
            v4 = v47 + 6;
            goto LABEL_3;
          }
        }
        while ( (unsigned int)v44 > 0x270F );
        v4 = v47 + 7;
      }
    }
  }
LABEL_3:
  v48 = v50;
  sub_2240A50((__int64 *)&v48, v4, 0);
  sub_2554A60(v48, v49, v3);
  v5 = (__m128i *)sub_2241130((unsigned __int64 *)&v48, 0, 0, "[AAExecutionDomain] ", 0x14u);
  v51 = &v53;
  if ( (__m128i *)v5->m128i_i64[0] == &v5[1] )
  {
    v53 = _mm_loadu_si128(v5 + 1);
  }
  else
  {
    v51 = (__m128i *)v5->m128i_i64[0];
    v53.m128i_i64[0] = v5[1].m128i_i64[0];
  }
  v6 = v5->m128i_i64[1];
  v5[1].m128i_i8[0] = 0;
  v52 = v6;
  v5->m128i_i64[0] = (__int64)v5[1].m128i_i64;
  v5->m128i_i64[1] = 0;
  if ( v52 == 0x3FFFFFFFFFFFFFFFLL )
LABEL_124:
    sub_4262D8((__int64)"basic_string::append");
  v7 = (__m128i *)sub_2241490((unsigned __int64 *)&v51, "/", 1u);
  v54 = &v56;
  if ( (__m128i *)v7->m128i_i64[0] == &v7[1] )
  {
    v56 = _mm_loadu_si128(v7 + 1);
  }
  else
  {
    v54 = (__m128i *)v7->m128i_i64[0];
    v56.m128i_i64[0] = v7[1].m128i_i64[0];
  }
  v8 = v7->m128i_u64[1];
  v7[1].m128i_i8[0] = 0;
  v55 = v8;
  v7->m128i_i64[0] = (__int64)v7[1].m128i_i64;
  v9 = v54;
  v7->m128i_i64[1] = 0;
  v10 = 15;
  v11 = 15;
  if ( v9 != &v56 )
    v11 = v56.m128i_i64[0];
  if ( v55 + v58 <= v11 )
    goto LABEL_14;
  if ( v57 != (char *)v59 )
    v10 = v59[0];
  if ( v55 + v58 <= v10 )
  {
    v12 = (__m128i *)sub_2241130((unsigned __int64 *)&v57, 0, 0, v9, v55);
    v60 = &v62;
    v13 = v12->m128i_i64[0];
    v14 = v12 + 1;
    if ( (__m128i *)v12->m128i_i64[0] != &v12[1] )
      goto LABEL_15;
  }
  else
  {
LABEL_14:
    v12 = (__m128i *)sub_2241490((unsigned __int64 *)&v54, v57, v58);
    v60 = &v62;
    v13 = v12->m128i_i64[0];
    v14 = v12 + 1;
    if ( (__m128i *)v12->m128i_i64[0] != &v12[1] )
    {
LABEL_15:
      v60 = (__m128i *)v13;
      v62.m128i_i64[0] = v12[1].m128i_i64[0];
      goto LABEL_16;
    }
  }
  v62 = _mm_loadu_si128(v12 + 1);
LABEL_16:
  v61 = v12->m128i_i64[1];
  v12->m128i_i64[0] = (__int64)v14;
  v12->m128i_i64[1] = 0;
  v12[1].m128i_i8[0] = 0;
  if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v61) <= 3 )
    goto LABEL_124;
  v15 = (__m128i *)sub_2241490((unsigned __int64 *)&v60, " of ", 4u);
  v63 = &v65;
  if ( (__m128i *)v15->m128i_i64[0] == &v15[1] )
  {
    v65 = _mm_loadu_si128(v15 + 1);
  }
  else
  {
    v63 = (__m128i *)v15->m128i_i64[0];
    v65.m128i_i64[0] = v15[1].m128i_i64[0];
  }
  v16 = v15->m128i_u64[1];
  v15[1].m128i_i8[0] = 0;
  v64 = v16;
  v15->m128i_i64[0] = (__int64)v15[1].m128i_i64;
  v17 = v63;
  v15->m128i_i64[1] = 0;
  v18 = 15;
  v19 = 15;
  if ( v17 != &v65 )
    v19 = v65.m128i_i64[0];
  if ( v64 + v67 <= v19 )
    goto LABEL_25;
  if ( v66 != (char *)v68 )
    v18 = v68[0];
  if ( v64 + v67 <= v18 )
  {
    v20 = (__m128i *)sub_2241130((unsigned __int64 *)&v66, 0, 0, v17, v64);
    v69 = v71;
    v21 = v20->m128i_i64[0];
    v22 = v20 + 1;
    if ( (__m128i *)v20->m128i_i64[0] != &v20[1] )
      goto LABEL_26;
  }
  else
  {
LABEL_25:
    v20 = (__m128i *)sub_2241490((unsigned __int64 *)&v63, v66, v67);
    v69 = v71;
    v21 = v20->m128i_i64[0];
    v22 = v20 + 1;
    if ( (__m128i *)v20->m128i_i64[0] != &v20[1] )
    {
LABEL_26:
      v69 = (_OWORD *)v21;
      *(_QWORD *)&v71[0] = v20[1].m128i_i64[0];
      goto LABEL_27;
    }
  }
  v71[0] = _mm_loadu_si128(v20 + 1);
LABEL_27:
  v70 = v20->m128i_i64[1];
  v20->m128i_i64[0] = (__int64)v22;
  v20->m128i_i64[1] = 0;
  v20[1].m128i_i8[0] = 0;
  if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v70) <= 0x24 )
    goto LABEL_124;
  v23 = (__m128i *)sub_2241490((unsigned __int64 *)&v69, " executed by initial thread / aligned", 0x25u);
  a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
  if ( (__m128i *)v23->m128i_i64[0] == &v23[1] )
  {
    a1[1] = _mm_loadu_si128(v23 + 1);
  }
  else
  {
    a1->m128i_i64[0] = v23->m128i_i64[0];
    a1[1].m128i_i64[0] = v23[1].m128i_i64[0];
  }
  v24 = v23->m128i_i64[1];
  v23->m128i_i64[0] = (__int64)v23[1].m128i_i64;
  v25 = v69;
  v23->m128i_i64[1] = 0;
  a1->m128i_i64[1] = v24;
  v23[1].m128i_i8[0] = 0;
  if ( v25 != v71 )
    j_j___libc_free_0((unsigned __int64)v25);
  if ( v63 != &v65 )
    j_j___libc_free_0((unsigned __int64)v63);
  if ( v60 != &v62 )
    j_j___libc_free_0((unsigned __int64)v60);
  if ( v54 != &v56 )
    j_j___libc_free_0((unsigned __int64)v54);
  if ( v51 != &v53 )
    j_j___libc_free_0((unsigned __int64)v51);
  if ( v48 != (_BYTE *)v50 )
    j_j___libc_free_0((unsigned __int64)v48);
  if ( v57 != (char *)v59 )
    j_j___libc_free_0((unsigned __int64)v57);
  if ( v66 != (char *)v68 )
    j_j___libc_free_0((unsigned __int64)v66);
  return a1;
}
