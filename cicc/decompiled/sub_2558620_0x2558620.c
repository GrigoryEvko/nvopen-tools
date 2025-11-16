// Function: sub_2558620
// Address: 0x2558620
//
__m128i *__fastcall sub_2558620(__m128i *a1, __int64 a2)
{
  __int64 v2; // rax
  char v3; // bl
  unsigned int v4; // r12d
  unsigned __int64 v5; // rdx
  unsigned int v6; // ecx
  unsigned __int64 v7; // rsi
  unsigned int v8; // r8d
  unsigned __int64 v9; // rsi
  __int64 v11; // r12
  char v12; // bl
  __int64 v13; // rdi
  __int64 v14; // r13
  __m128i *v15; // rax
  unsigned __int64 v16; // rcx
  __int64 v17; // r15
  __int64 *i; // rax
  unsigned __int64 v19; // r12
  unsigned __int64 v20; // r9
  unsigned __int64 v21; // rdx
  int v22; // r14d
  unsigned __int64 v23; // rcx
  int v24; // eax
  __int64 v25; // r12
  __int64 *v26; // rax
  unsigned __int64 v27; // r9
  unsigned __int64 v28; // r8
  unsigned __int64 v29; // rdx
  int v30; // r15d
  unsigned __int64 v31; // rcx
  int v32; // eax
  __int64 *v33; // rax
  unsigned __int64 v34; // r11
  unsigned __int64 v35; // r9
  unsigned __int64 v36; // rdx
  int v37; // r8d
  unsigned __int64 v38; // rcx
  int v39; // eax
  __int64 *v40; // [rsp+0h] [rbp-190h]
  __int64 v42; // [rsp+18h] [rbp-178h]
  unsigned __int64 v43; // [rsp+20h] [rbp-170h]
  unsigned __int64 v44; // [rsp+28h] [rbp-168h]
  unsigned __int64 v45; // [rsp+28h] [rbp-168h]
  unsigned __int64 v46; // [rsp+28h] [rbp-168h]
  unsigned __int64 v47; // [rsp+30h] [rbp-160h]
  unsigned __int64 v48; // [rsp+30h] [rbp-160h]
  int v49; // [rsp+30h] [rbp-160h]
  unsigned __int64 v50[4]; // [rsp+40h] [rbp-150h] BYREF
  unsigned __int64 v51[4]; // [rsp+60h] [rbp-130h] BYREF
  char *v52; // [rsp+80h] [rbp-110h] BYREF
  int v53; // [rsp+88h] [rbp-108h]
  char v54; // [rsp+90h] [rbp-100h] BYREF
  __m128i v55[2]; // [rsp+A0h] [rbp-F0h] BYREF
  __m128i v56[2]; // [rsp+C0h] [rbp-D0h] BYREF
  __m128i v57[2]; // [rsp+E0h] [rbp-B0h] BYREF
  char *v58; // [rsp+100h] [rbp-90h] BYREF
  __int64 v59; // [rsp+108h] [rbp-88h]
  char v60; // [rsp+110h] [rbp-80h] BYREF
  unsigned __int64 v61[2]; // [rsp+120h] [rbp-70h] BYREF
  __m128i v62; // [rsp+130h] [rbp-60h] BYREF
  __m128i v63; // [rsp+140h] [rbp-50h] BYREF
  _QWORD v64[8]; // [rsp+150h] [rbp-40h] BYREF

  if ( *(_QWORD *)(a2 + 376) )
  {
    v11 = *(_QWORD *)(a2 + 360);
    v42 = a2 + 344;
    v12 = 0;
  }
  else
  {
    v2 = *(unsigned int *)(a2 + 296);
    if ( !(_DWORD)v2 )
    {
      v3 = 0;
      sub_253C590(v63.m128i_i64, byte_3F871B3);
      goto LABEL_4;
    }
    v11 = *(_QWORD *)(a2 + 288);
    v12 = 1;
    v42 = v11 + 8 * v2;
  }
  v60 = 0;
  v58 = &v60;
  v59 = 0;
  if ( v11 == v42 )
    goto LABEL_28;
  v13 = v11;
  v14 = 0;
  while ( !v12 )
  {
    if ( v13 == v42 )
      goto LABEL_31;
    v13 = sub_220EF30(v13);
LABEL_24:
    ++v14;
  }
  if ( v13 != v42 )
  {
    v13 += 8;
    goto LABEL_24;
  }
LABEL_31:
  v40 = (__int64 *)v11;
  v17 = v11;
  v44 = 2 * v14 - 2;
  if ( v12 )
    goto LABEL_47;
LABEL_32:
  if ( v17 == v42 )
  {
    v25 = (__int64)v40;
    sub_2240E30((__int64)&v58, v44);
    v26 = v40 + 4;
  }
  else
  {
    for ( i = (__int64 *)(v17 + 32); ; i = (__int64 *)v17 )
    {
      v19 = (unsigned __int64)*i >> 63;
      v20 = abs64(*i);
      if ( v20 <= 9 )
      {
        v22 = 1;
      }
      else if ( v20 <= 0x63 )
      {
        v22 = 2;
      }
      else if ( v20 <= 0x3E7 )
      {
        v22 = 3;
      }
      else if ( v20 <= 0x270F )
      {
        v22 = 4;
      }
      else
      {
        v21 = v20;
        v22 = 1;
        while ( 1 )
        {
          v23 = v21;
          v24 = v22;
          v22 += 4;
          v21 /= 0x2710u;
          if ( v23 <= 0x1869F )
            break;
          if ( v23 <= 0xF423F )
          {
            v22 = v24 + 5;
            break;
          }
          if ( v23 <= (unsigned __int64)&loc_98967F )
          {
            v22 = v24 + 6;
            break;
          }
          if ( v23 <= 0x5F5E0FF )
          {
            v22 = v24 + 7;
            break;
          }
        }
      }
      v47 = v20;
      v63.m128i_i64[0] = (__int64)v64;
      sub_2240A50(v63.m128i_i64, (unsigned int)(v22 + v19), 45);
      sub_1249540((_BYTE *)(v19 + v63.m128i_i64[0]), v22, v47);
      v44 += v63.m128i_u64[1];
      if ( (_QWORD *)v63.m128i_i64[0] != v64 )
        j_j___libc_free_0(v63.m128i_u64[0]);
      if ( !v12 )
      {
        v17 = sub_220EF30(v17);
        goto LABEL_32;
      }
      v17 += 8;
LABEL_47:
      if ( v17 == v42 )
        break;
    }
    v25 = (__int64)v40;
    sub_2240E30((__int64)&v58, v44);
    v26 = v40;
  }
  v27 = (unsigned __int64)*v26 >> 63;
  v28 = abs64(*v26);
  if ( v28 <= 9 )
  {
    v30 = 1;
  }
  else if ( v28 <= 0x63 )
  {
    v30 = 2;
  }
  else if ( v28 <= 0x3E7 )
  {
    v30 = 3;
  }
  else if ( v28 <= 0x270F )
  {
    v30 = 4;
  }
  else
  {
    v29 = v28;
    v30 = 1;
    while ( 1 )
    {
      v31 = v29;
      v32 = v30;
      v30 += 4;
      v29 /= 0x2710u;
      if ( v31 <= 0x1869F )
        break;
      if ( v31 <= 0xF423F )
      {
        v30 = v32 + 5;
        break;
      }
      if ( v31 <= (unsigned __int64)&loc_98967F )
      {
        v30 = v32 + 6;
        break;
      }
      if ( v31 <= 0x5F5E0FF )
      {
        v30 = v32 + 7;
        break;
      }
    }
  }
  v45 = v28;
  v63.m128i_i64[0] = (__int64)v64;
  v48 = v27;
  sub_2240A50(v63.m128i_i64, (unsigned int)(v30 + v27), 45);
  sub_1249540((_BYTE *)(v48 + v63.m128i_i64[0]), v30, v45);
  sub_2241490((unsigned __int64 *)&v58, (char *)v63.m128i_i64[0], v63.m128i_u64[1]);
  sub_2240A30((unsigned __int64 *)&v63);
LABEL_65:
  if ( v12 )
  {
LABEL_66:
    v25 += 8;
    if ( v25 != v42 )
      goto LABEL_67;
  }
  else
  {
    while ( 1 )
    {
      v25 = sub_220EF30(v25);
      if ( v25 == v42 )
        break;
LABEL_67:
      if ( v59 == 0x3FFFFFFFFFFFFFFFLL || v59 == 4611686018427387902LL )
        sub_4262D8((__int64)"basic_string::append");
      sub_2241490((unsigned __int64 *)&v58, ", ", 2u);
      v33 = (__int64 *)(v25 + 32);
      if ( v12 )
        v33 = (__int64 *)v25;
      v34 = (unsigned __int64)*v33 >> 63;
      v35 = abs64(*v33);
      if ( v35 <= 9 )
      {
        v37 = 1;
      }
      else if ( v35 <= 0x63 )
      {
        v37 = 2;
      }
      else if ( v35 <= 0x3E7 )
      {
        v37 = 3;
      }
      else if ( v35 <= 0x270F )
      {
        v37 = 4;
      }
      else
      {
        v36 = v35;
        v37 = 1;
        while ( 1 )
        {
          v38 = v36;
          v39 = v37;
          v37 += 4;
          v36 /= 0x2710u;
          if ( v38 <= 0x1869F )
            break;
          if ( v38 <= 0xF423F )
          {
            v37 = v39 + 5;
            break;
          }
          if ( v38 <= (unsigned __int64)&loc_98967F )
          {
            v37 = v39 + 6;
            break;
          }
          if ( v38 <= 0x5F5E0FF )
          {
            v37 = v39 + 7;
            break;
          }
        }
      }
      v43 = v35;
      v63.m128i_i64[0] = (__int64)v64;
      v46 = v34;
      v49 = v37;
      sub_2240A50(v63.m128i_i64, (unsigned int)(v37 + v34), 45);
      sub_1249540((_BYTE *)(v46 + v63.m128i_i64[0]), v49, v43);
      sub_2241490((unsigned __int64 *)&v58, (char *)v63.m128i_i64[0], v63.m128i_u64[1]);
      if ( (_QWORD *)v63.m128i_i64[0] == v64 )
        goto LABEL_65;
      j_j___libc_free_0(v63.m128i_u64[0]);
      if ( v12 )
        goto LABEL_66;
    }
  }
LABEL_28:
  v15 = (__m128i *)sub_2241130((unsigned __int64 *)&v58, 0, 0, " (returned:", 0xBu);
  v61[0] = (unsigned __int64)&v62;
  if ( (__m128i *)v15->m128i_i64[0] == &v15[1] )
  {
    v62 = _mm_loadu_si128(v15 + 1);
  }
  else
  {
    v61[0] = v15->m128i_i64[0];
    v62.m128i_i64[0] = v15[1].m128i_i64[0];
  }
  v16 = v15->m128i_u64[1];
  v15[1].m128i_i8[0] = 0;
  v3 = 1;
  v61[1] = v16;
  v15->m128i_i64[0] = (__int64)v15[1].m128i_i64;
  v15->m128i_i64[1] = 0;
  sub_94F930(&v63, (__int64)v61, ")");
LABEL_4:
  if ( *(_BYTE *)(a2 + 393) )
  {
    v4 = *(_DWORD *)(a2 + 240);
    if ( v4 <= 9 )
    {
      v9 = 1;
    }
    else if ( v4 <= 0x63 )
    {
      v9 = 2;
    }
    else if ( v4 <= 0x3E7 )
    {
      v9 = 3;
    }
    else
    {
      v5 = v4;
      if ( v4 <= 0x270F )
      {
        v9 = 4;
      }
      else
      {
        v6 = 1;
        do
        {
          v7 = v5;
          v8 = v6;
          v6 += 4;
          v5 /= 0x2710u;
          if ( v7 <= 0x1869F )
          {
            v9 = v6;
            goto LABEL_15;
          }
          if ( (unsigned int)v5 <= 0x63 )
          {
            v9 = v8 + 5;
            goto LABEL_15;
          }
          if ( (unsigned int)v5 <= 0x3E7 )
          {
            v9 = v8 + 6;
            goto LABEL_15;
          }
        }
        while ( (unsigned int)v5 > 0x270F );
        v9 = v8 + 7;
      }
    }
LABEL_15:
    v52 = &v54;
    sub_2240A50((__int64 *)&v52, v9, 0);
    sub_2554A60(v52, v53, v4);
    sub_253C590((__int64 *)v51, "#");
    sub_8FD5D0(v55, (__int64)v51, &v52);
    sub_94F930(v56, (__int64)v55, " bins");
    sub_253C590((__int64 *)v50, "PointerInfo ");
    sub_8FD5D0(v57, (__int64)v50, v56);
    sub_8FD5D0(a1, (__int64)v57, &v63);
    sub_2240A30((unsigned __int64 *)v57);
    sub_2240A30(v50);
    sub_2240A30((unsigned __int64 *)v56);
    sub_2240A30((unsigned __int64 *)v55);
    sub_2240A30(v51);
    sub_2240A30((unsigned __int64 *)&v52);
    sub_2240A30((unsigned __int64 *)&v63);
    if ( v3 )
    {
LABEL_16:
      sub_2240A30(v61);
      sub_2240A30((unsigned __int64 *)&v58);
    }
  }
  else
  {
    sub_253C590(v56[0].m128i_i64, "<invalid>");
    sub_253C590((__int64 *)v50, "PointerInfo ");
    sub_8FD5D0(v57, (__int64)v50, v56);
    sub_8FD5D0(a1, (__int64)v57, &v63);
    sub_2240A30((unsigned __int64 *)v57);
    sub_2240A30(v50);
    sub_2240A30((unsigned __int64 *)v56);
    sub_2240A30((unsigned __int64 *)&v63);
    if ( v3 )
      goto LABEL_16;
  }
  return a1;
}
