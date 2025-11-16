// Function: sub_25573D0
// Address: 0x25573d0
//
__m128i *__fastcall sub_25573D0(__m128i *a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // rbx
  unsigned __int64 v3; // r13
  unsigned __int64 v4; // rdx
  unsigned __int64 v5; // rcx
  int v6; // eax
  unsigned __int64 v7; // r14
  unsigned __int64 v8; // rdx
  unsigned __int64 v9; // rsi
  unsigned __int64 v10; // rcx
  int v11; // eax
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  unsigned __int64 v15; // r15
  unsigned __int64 v16; // rdx
  unsigned __int64 v17; // rsi
  unsigned __int64 v18; // rcx
  int v19; // eax
  unsigned int v20; // ebx
  unsigned __int64 v21; // rdx
  unsigned __int64 v22; // rsi
  unsigned __int64 v23; // rcx
  int v24; // r8d
  __m128i *v25; // rax
  unsigned __int64 v26; // rcx
  char *v28; // [rsp+40h] [rbp-190h] BYREF
  int v29; // [rsp+48h] [rbp-188h]
  char v30; // [rsp+50h] [rbp-180h] BYREF
  unsigned __int64 v31[2]; // [rsp+60h] [rbp-170h] BYREF
  __m128i v32; // [rsp+70h] [rbp-160h] BYREF
  __m128i v33[2]; // [rsp+80h] [rbp-150h] BYREF
  char *v34; // [rsp+A0h] [rbp-130h] BYREF
  int v35; // [rsp+A8h] [rbp-128h]
  char v36; // [rsp+B0h] [rbp-120h] BYREF
  __m128i v37[2]; // [rsp+C0h] [rbp-110h] BYREF
  __m128i v38[2]; // [rsp+E0h] [rbp-F0h] BYREF
  char *v39; // [rsp+100h] [rbp-D0h] BYREF
  int v40; // [rsp+108h] [rbp-C8h]
  char v41; // [rsp+110h] [rbp-C0h] BYREF
  __m128i v42[2]; // [rsp+120h] [rbp-B0h] BYREF
  __m128i v43[2]; // [rsp+140h] [rbp-90h] BYREF
  char *v44; // [rsp+160h] [rbp-70h] BYREF
  int v45; // [rsp+168h] [rbp-68h]
  char v46; // [rsp+170h] [rbp-60h] BYREF
  __m128i v47[5]; // [rsp+180h] [rbp-50h] BYREF

  v2 = a2;
  v3 = *(unsigned int *)(a2 + 256);
  if ( v3 <= 9 )
  {
    a2 = 1;
  }
  else if ( v3 <= 0x63 )
  {
    a2 = 2;
  }
  else if ( v3 <= 0x3E7 )
  {
    a2 = 3;
  }
  else if ( v3 <= 0x270F )
  {
    a2 = 4;
  }
  else
  {
    v4 = *(unsigned int *)(a2 + 256);
    LODWORD(a2) = 1;
    while ( 1 )
    {
      v5 = v4;
      v6 = a2;
      a2 = (unsigned int)(a2 + 4);
      v4 /= 0x2710u;
      if ( v5 <= 0x1869F )
        break;
      if ( v5 <= 0xF423F )
      {
        a2 = (unsigned int)(v6 + 5);
        break;
      }
      if ( v5 <= (unsigned __int64)&loc_98967F )
      {
        a2 = (unsigned int)(v6 + 6);
        break;
      }
      if ( v5 <= 0x5F5E0FF )
      {
        a2 = (unsigned int)(v6 + 7);
        break;
      }
    }
  }
  v44 = &v46;
  sub_2240A50((__int64 *)&v44, a2, 0);
  sub_1249540(v44, v45, v3);
  v7 = *(unsigned int *)(v2 + 144);
  if ( v7 <= 9 )
  {
    v9 = 1;
  }
  else if ( v7 <= 0x63 )
  {
    v9 = 2;
  }
  else if ( v7 <= 0x3E7 )
  {
    v9 = 3;
  }
  else if ( v7 <= 0x270F )
  {
    v9 = 4;
  }
  else
  {
    v8 = *(unsigned int *)(v2 + 144);
    LODWORD(v9) = 1;
    while ( 1 )
    {
      v10 = v8;
      v11 = v9;
      v9 = (unsigned int)(v9 + 4);
      v8 /= 0x2710u;
      if ( v10 <= 0x1869F )
        break;
      if ( v10 <= 0xF423F )
      {
        v9 = (unsigned int)(v11 + 5);
        break;
      }
      if ( v10 <= (unsigned __int64)&loc_98967F )
      {
        v9 = (unsigned int)(v11 + 6);
        break;
      }
      if ( v10 <= 0x5F5E0FF )
      {
        v9 = (unsigned int)(v11 + 7);
        break;
      }
    }
  }
  v39 = &v41;
  sub_2240A50((__int64 *)&v39, v9, 0);
  sub_1249540(v39, v40, v7);
  v12 = sub_25096F0((_QWORD *)(v2 + 72));
  v13 = v12 + 72;
  v14 = *(_QWORD *)(v12 + 80);
  if ( v14 == v13 )
  {
    v17 = 1;
    v15 = 0;
  }
  else
  {
    v15 = 0;
    do
    {
      v14 = *(_QWORD *)(v14 + 8);
      ++v15;
    }
    while ( v13 != v14 );
    if ( (__int64)v15 <= 9 )
    {
      v17 = 1;
    }
    else if ( v15 <= 0x63 )
    {
      v17 = 2;
    }
    else if ( v15 <= 0x3E7 )
    {
      v17 = 3;
    }
    else if ( v15 <= 0x270F )
    {
      v17 = 4;
    }
    else
    {
      v16 = v15;
      LODWORD(v17) = 1;
      while ( 1 )
      {
        v18 = v16;
        v19 = v17;
        v17 = (unsigned int)(v17 + 4);
        v16 /= 0x2710u;
        if ( v18 <= 0x1869F )
          break;
        if ( v18 <= 0xF423F )
        {
          v17 = (unsigned int)(v19 + 5);
          break;
        }
        if ( v18 <= (unsigned __int64)&loc_98967F )
        {
          v17 = (unsigned int)(v19 + 6);
          break;
        }
        if ( v18 <= 0x5F5E0FF )
        {
          v17 = (unsigned int)(v19 + 7);
          break;
        }
      }
    }
  }
  v34 = &v36;
  sub_2240A50((__int64 *)&v34, v17, 0);
  sub_1249540(v34, v35, v15);
  v20 = *(_DWORD *)(v2 + 376);
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
  else
  {
    v21 = v20;
    if ( v20 <= 0x270F )
    {
      v22 = 4;
    }
    else
    {
      LODWORD(v22) = 1;
      while ( 1 )
      {
        v23 = v21;
        v24 = v22;
        v22 = (unsigned int)(v22 + 4);
        v21 /= 0x2710u;
        if ( v23 <= 0x1869F )
          break;
        if ( (unsigned int)v21 <= 0x63 )
        {
          v22 = (unsigned int)(v24 + 5);
          break;
        }
        if ( (unsigned int)v21 <= 0x3E7 )
        {
          v22 = (unsigned int)(v24 + 6);
          break;
        }
        if ( (unsigned int)v21 <= 0x270F )
        {
          v22 = (unsigned int)(v24 + 7);
          break;
        }
      }
    }
  }
  v28 = &v30;
  sub_2240A50((__int64 *)&v28, v22, 0);
  sub_2554A60(v28, v29, v20);
  v25 = (__m128i *)sub_2241130((unsigned __int64 *)&v28, 0, 0, "Live[#BB ", 9u);
  v31[0] = (unsigned __int64)&v32;
  if ( (__m128i *)v25->m128i_i64[0] == &v25[1] )
  {
    v32 = _mm_loadu_si128(v25 + 1);
  }
  else
  {
    v31[0] = v25->m128i_i64[0];
    v32.m128i_i64[0] = v25[1].m128i_i64[0];
  }
  v26 = v25->m128i_u64[1];
  v25[1].m128i_i8[0] = 0;
  v31[1] = v26;
  v25->m128i_i64[0] = (__int64)v25[1].m128i_i64;
  v25->m128i_i64[1] = 0;
  sub_94F930(v33, (__int64)v31, "/");
  sub_8FD5D0(v37, (__int64)v33, &v34);
  sub_94F930(v38, (__int64)v37, "][#TBEP ");
  sub_8FD5D0(v42, (__int64)v38, &v39);
  sub_94F930(v43, (__int64)v42, "][#KDE ");
  sub_8FD5D0(v47, (__int64)v43, &v44);
  sub_94F930(a1, (__int64)v47, "]");
  sub_2240A30((unsigned __int64 *)v47);
  sub_2240A30((unsigned __int64 *)v43);
  sub_2240A30((unsigned __int64 *)v42);
  sub_2240A30((unsigned __int64 *)v38);
  sub_2240A30((unsigned __int64 *)v37);
  sub_2240A30((unsigned __int64 *)v33);
  sub_2240A30(v31);
  sub_2240A30((unsigned __int64 *)&v28);
  sub_2240A30((unsigned __int64 *)&v34);
  sub_2240A30((unsigned __int64 *)&v39);
  sub_2240A30((unsigned __int64 *)&v44);
  return a1;
}
