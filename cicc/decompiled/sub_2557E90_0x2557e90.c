// Function: sub_2557E90
// Address: 0x2557e90
//
__m128i *__fastcall sub_2557E90(__m128i *a1, __int64 a2)
{
  unsigned int v2; // r15d
  unsigned int v3; // r14d
  __int64 v4; // rax
  __int64 v5; // rcx
  unsigned __int64 v6; // rdx
  unsigned __int64 v7; // rsi
  unsigned __int64 v8; // rcx
  int v9; // r8d
  unsigned __int64 v10; // rdx
  unsigned __int64 v11; // rcx
  int v12; // r8d
  __m128i *v13; // rax
  unsigned __int64 v14; // rcx
  unsigned __int64 v16; // rsi
  _BYTE *v17; // [rsp+10h] [rbp-B0h] BYREF
  int v18; // [rsp+18h] [rbp-A8h]
  _QWORD v19[2]; // [rsp+20h] [rbp-A0h] BYREF
  unsigned __int64 v20[2]; // [rsp+30h] [rbp-90h] BYREF
  __m128i v21; // [rsp+40h] [rbp-80h] BYREF
  __m128i v22; // [rsp+50h] [rbp-70h] BYREF
  __int64 v23; // [rsp+60h] [rbp-60h] BYREF
  _BYTE *v24; // [rsp+70h] [rbp-50h] BYREF
  int v25; // [rsp+78h] [rbp-48h]
  _QWORD v26[8]; // [rsp+80h] [rbp-40h] BYREF

  v2 = 0;
  v3 = 0;
  v4 = *(_QWORD *)(a2 + 136);
  v5 = v4 + 16LL * *(unsigned int *)(a2 + 144);
  if ( v5 == v4 )
  {
    v24 = v26;
    sub_2240A50((__int64 *)&v24, 1u, 0);
    sub_2554A60(v24, v25, 0);
    v16 = 1;
  }
  else
  {
    do
    {
      while ( *(_DWORD *)(*(_QWORD *)(v4 + 8) + 12LL) != 2 )
      {
        v4 += 16;
        ++v3;
        if ( v4 == v5 )
          goto LABEL_6;
      }
      v4 += 16;
      ++v2;
    }
    while ( v4 != v5 );
LABEL_6:
    if ( v2 <= 9 )
    {
      v7 = 1;
    }
    else if ( v2 <= 0x63 )
    {
      v7 = 2;
    }
    else if ( v2 <= 0x3E7 )
    {
      v7 = 3;
    }
    else
    {
      v6 = v2;
      if ( v2 <= 0x270F )
      {
        v7 = 4;
      }
      else
      {
        LODWORD(v7) = 1;
        while ( 1 )
        {
          v8 = v6;
          v9 = v7;
          v7 = (unsigned int)(v7 + 4);
          v6 /= 0x2710u;
          if ( v8 <= 0x1869F )
            break;
          if ( (unsigned int)v6 <= 0x63 )
          {
            v7 = (unsigned int)(v9 + 5);
            break;
          }
          if ( (unsigned int)v6 <= 0x3E7 )
          {
            v7 = (unsigned int)(v9 + 6);
            break;
          }
          if ( (unsigned int)v6 <= 0x270F )
          {
            v7 = (unsigned int)(v9 + 7);
            break;
          }
        }
      }
    }
    v24 = v26;
    sub_2240A50((__int64 *)&v24, v7, 0);
    sub_2554A60(v24, v25, v2);
    if ( v3 <= 9 )
    {
      v16 = 1;
    }
    else if ( v3 <= 0x63 )
    {
      v16 = 2;
    }
    else if ( v3 <= 0x3E7 )
    {
      v16 = 3;
    }
    else
    {
      v10 = v3;
      if ( v3 <= 0x270F )
      {
        v16 = 4;
      }
      else
      {
        LODWORD(v16) = 1;
        while ( 1 )
        {
          v11 = v10;
          v12 = v16;
          v16 = (unsigned int)(v16 + 4);
          v10 /= 0x2710u;
          if ( v11 <= 0x1869F )
            break;
          if ( (unsigned int)v10 <= 0x63 )
          {
            v16 = (unsigned int)(v12 + 5);
            break;
          }
          if ( (unsigned int)v10 <= 0x3E7 )
          {
            v16 = (unsigned int)(v12 + 6);
            break;
          }
          if ( (unsigned int)v10 <= 0x270F )
          {
            v16 = (unsigned int)(v12 + 7);
            break;
          }
        }
      }
    }
  }
  v17 = v19;
  sub_2240A50((__int64 *)&v17, v16, 0);
  sub_2554A60(v17, v18, v3);
  v13 = (__m128i *)sub_2241130((unsigned __int64 *)&v17, 0, 0, "[H2S] Mallocs Good/Bad: ", 0x18u);
  v20[0] = (unsigned __int64)&v21;
  if ( (__m128i *)v13->m128i_i64[0] == &v13[1] )
  {
    v21 = _mm_loadu_si128(v13 + 1);
  }
  else
  {
    v20[0] = v13->m128i_i64[0];
    v21.m128i_i64[0] = v13[1].m128i_i64[0];
  }
  v14 = v13->m128i_u64[1];
  v13[1].m128i_i8[0] = 0;
  v20[1] = v14;
  v13->m128i_i64[0] = (__int64)v13[1].m128i_i64;
  v13->m128i_i64[1] = 0;
  sub_94F930(&v22, (__int64)v20, "/");
  sub_8FD5D0(a1, (__int64)&v22, &v24);
  if ( (__int64 *)v22.m128i_i64[0] != &v23 )
    j_j___libc_free_0(v22.m128i_u64[0]);
  if ( (__m128i *)v20[0] != &v21 )
    j_j___libc_free_0(v20[0]);
  if ( v17 != (_BYTE *)v19 )
    j_j___libc_free_0((unsigned __int64)v17);
  if ( v24 != (_BYTE *)v26 )
    j_j___libc_free_0((unsigned __int64)v24);
  return a1;
}
