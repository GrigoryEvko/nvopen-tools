// Function: sub_BAD140
// Address: 0xbad140
//
__int64 __fastcall sub_BAD140(__int64 a1, _QWORD *a2, unsigned __int64 a3, unsigned __int64 a4, __int64 a5)
{
  unsigned __int64 v5; // r14
  __int64 v7; // r12
  __int64 v8; // rsi
  int v9; // eax
  _BYTE *v10; // r8
  unsigned __int64 v11; // rcx
  int v12; // esi
  __int64 v13; // rdx
  unsigned __int64 v14; // rax
  char v15; // r11
  __int64 v16; // r10
  _WORD *v17; // rdx
  __int64 v18; // rax
  unsigned __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // rdx
  unsigned __int64 v22; // rdx
  __int64 v23; // rsi
  int v24; // eax
  _BYTE *v25; // rsi
  unsigned __int32 v26; // ecx
  __int64 v27; // rdx
  unsigned __int64 v28; // rax
  char v29; // r10
  __int64 v30; // r9
  __m128i *v31; // rax
  __m128i *v32; // rsi
  __m128i *v33; // rdx
  __m128i *v34; // rdx
  __m128i si128; // xmm0
  __int64 result; // rax
  _QWORD *v37; // [rsp+10h] [rbp-B0h] BYREF
  __int64 v38; // [rsp+18h] [rbp-A8h]
  _QWORD v39[2]; // [rsp+20h] [rbp-A0h] BYREF
  __m128i *v40; // [rsp+30h] [rbp-90h]
  __int64 v41; // [rsp+38h] [rbp-88h]
  __m128i v42; // [rsp+40h] [rbp-80h] BYREF
  _QWORD v43[2]; // [rsp+50h] [rbp-70h] BYREF
  _QWORD v44[2]; // [rsp+60h] [rbp-60h] BYREF
  __m128i v45; // [rsp+70h] [rbp-50h] BYREF
  _QWORD v46[8]; // [rsp+80h] [rbp-40h] BYREF

  v5 = a3;
  v7 = a1;
  if ( a3 <= 9 )
  {
    v37 = v39;
    sub_2240A50(&v37, 1, 0, a4, a5);
    v10 = v37;
    LOBYTE(v11) = v5;
LABEL_14:
    *v10 = v11 + 48;
    goto LABEL_15;
  }
  if ( a3 <= 0x63 )
  {
    v37 = v39;
    sub_2240A50(&v37, 2, 0, a4, a5);
    v10 = v37;
    v11 = v5;
  }
  else
  {
    if ( a3 <= 0x3E7 )
    {
      v8 = 3;
    }
    else if ( a3 <= 0x270F )
    {
      v8 = 4;
    }
    else
    {
      LODWORD(v8) = 1;
      while ( 1 )
      {
        a4 = a3;
        v9 = v8;
        v8 = (unsigned int)(v8 + 4);
        a3 /= 0x2710u;
        if ( a4 <= 0x1869F )
          break;
        if ( a4 <= 0xF423F )
        {
          v37 = v39;
          v8 = (unsigned int)(v9 + 5);
          goto LABEL_11;
        }
        if ( a4 <= (unsigned __int64)&loc_98967F )
        {
          v8 = (unsigned int)(v9 + 6);
          break;
        }
        if ( a4 <= 0x5F5E0FF )
        {
          v8 = (unsigned int)(v9 + 7);
          break;
        }
      }
    }
    v37 = v39;
LABEL_11:
    sub_2240A50(&v37, v8, 0, a4, a5);
    v10 = v37;
    v11 = v5;
    v12 = v38 - 1;
    do
    {
      v13 = v11
          - 20 * (v11 / 0x64 + (((0x28F5C28F5C28F5C3LL * (unsigned __int128)(v11 >> 2)) >> 64) & 0xFFFFFFFFFFFFFFFCLL));
      v14 = v11;
      v11 /= 0x64u;
      v15 = a00010203040506_0[2 * v13 + 1];
      LOBYTE(v13) = a00010203040506_0[2 * v13];
      v10[v12] = v15;
      v16 = (unsigned int)(v12 - 1);
      v12 -= 2;
      v10[v16] = v13;
    }
    while ( v14 > 0x270F );
    if ( v14 <= 0x3E7 )
      goto LABEL_14;
  }
  v10[1] = a00010203040506_0[2 * v11 + 1];
  *v10 = a00010203040506_0[2 * v11];
LABEL_15:
  v17 = *(_WORD **)(a1 + 32);
  if ( *(_QWORD *)(a1 + 24) - (_QWORD)v17 <= 1u )
  {
    a1 = sub_CB6200(a1, "  ", 2);
  }
  else
  {
    *v17 = 8224;
    *(_QWORD *)(a1 + 32) += 2LL;
  }
  v18 = sub_CB6200(a1, v37, v38);
  v21 = *(_QWORD *)(v18 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(v18 + 24) - v21) <= 8 )
  {
    sub_CB6200(v18, " [label=\"", 9);
    if ( (*a2 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    {
LABEL_19:
      if ( v5 > 9 )
      {
        if ( v5 <= 0x63 )
        {
          v45.m128i_i64[0] = (__int64)v46;
          sub_2240A50(&v45, 2, 0, v19, v20);
          v25 = (_BYTE *)v45.m128i_i64[0];
        }
        else
        {
          if ( v5 <= 0x3E7 )
          {
            v23 = 3;
          }
          else if ( v5 <= 0x270F )
          {
            v23 = 4;
          }
          else
          {
            v22 = v5;
            LODWORD(v23) = 1;
            while ( 1 )
            {
              v19 = v22;
              v24 = v23;
              v23 = (unsigned int)(v23 + 4);
              v22 /= 0x2710u;
              if ( v19 <= 0x1869F )
                break;
              if ( v19 <= 0xF423F )
              {
                v45.m128i_i64[0] = (__int64)v46;
                v23 = (unsigned int)(v24 + 5);
                goto LABEL_29;
              }
              if ( v19 <= (unsigned __int64)&loc_98967F )
              {
                v23 = (unsigned int)(v24 + 6);
                break;
              }
              if ( v19 <= 0x5F5E0FF )
              {
                v23 = (unsigned int)(v24 + 7);
                break;
              }
            }
          }
          v45.m128i_i64[0] = (__int64)v46;
LABEL_29:
          sub_2240A50(&v45, v23, 0, v19, v20);
          v25 = (_BYTE *)v45.m128i_i64[0];
          v26 = v45.m128i_i32[2] - 1;
          do
          {
            v27 = v5
                - 20
                * (v5 / 0x64 + (((0x28F5C28F5C28F5C3LL * (unsigned __int128)(v5 >> 2)) >> 64) & 0xFFFFFFFFFFFFFFFCLL));
            v28 = v5;
            v5 /= 0x64u;
            v29 = a00010203040506_0[2 * v27 + 1];
            LOBYTE(v27) = a00010203040506_0[2 * v27];
            v25[v26] = v29;
            v30 = v26 - 1;
            v26 -= 2;
            v25[v30] = v27;
          }
          while ( v28 > 0x270F );
          if ( v28 <= 0x3E7 )
            goto LABEL_32;
        }
        v25[1] = a00010203040506_0[2 * v5 + 1];
        *v25 = a00010203040506_0[2 * v5];
        goto LABEL_33;
      }
      v45.m128i_i64[0] = (__int64)v46;
      sub_2240A50(&v45, 1, 0, v19, v20);
      v25 = (_BYTE *)v45.m128i_i64[0];
LABEL_32:
      *v25 = v5 + 48;
LABEL_33:
      v43[1] = 1;
      LOWORD(v44[0]) = 64;
      v43[0] = v44;
      if ( (unsigned __int64)(v45.m128i_i64[1] + 1) <= 0xF
        || (_QWORD *)v45.m128i_i64[0] == v46
        || (unsigned __int64)(v45.m128i_i64[1] + 1) > v46[0] )
      {
        v31 = (__m128i *)sub_2241490(v43, v45.m128i_i64[0], v45.m128i_i64[1], v44);
        v40 = &v42;
        v32 = (__m128i *)v31->m128i_i64[0];
        v33 = v31 + 1;
        if ( (__m128i *)v31->m128i_i64[0] != &v31[1] )
        {
LABEL_37:
          v40 = v32;
          v42.m128i_i64[0] = v31[1].m128i_i64[0];
LABEL_38:
          v41 = v31->m128i_i64[1];
          v31->m128i_i64[0] = (__int64)v33;
          v31->m128i_i64[1] = 0;
          v31[1].m128i_i8[0] = 0;
          if ( (_QWORD *)v43[0] != v44 )
            j_j___libc_free_0(v43[0], v44[0] + 1LL);
          if ( (_QWORD *)v45.m128i_i64[0] != v46 )
            j_j___libc_free_0(v45.m128i_i64[0], v46[0] + 1LL);
          sub_CB6200(v7, v40, v41);
          if ( v40 != &v42 )
            j_j___libc_free_0(v40, v42.m128i_i64[0] + 1);
          goto LABEL_44;
        }
      }
      else
      {
        v31 = (__m128i *)sub_2241130(&v45, 0, 0, v44, 1);
        v40 = &v42;
        v32 = (__m128i *)v31->m128i_i64[0];
        v33 = v31 + 1;
        if ( (__m128i *)v31->m128i_i64[0] != &v31[1] )
          goto LABEL_37;
      }
      v42 = _mm_loadu_si128(v31 + 1);
      goto LABEL_38;
    }
  }
  else
  {
    *(_BYTE *)(v21 + 8) = 34;
    *(_QWORD *)v21 = 0x3D6C6562616C5B20LL;
    *(_QWORD *)(v18 + 32) += 9LL;
    if ( (*a2 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
      goto LABEL_19;
  }
  sub_BAC460(&v45, a2, v21, v19, v20);
  sub_CB6200(v7, v45.m128i_i64[0], v45.m128i_i64[1]);
  if ( (_QWORD *)v45.m128i_i64[0] != v46 )
    j_j___libc_free_0(v45.m128i_i64[0], v46[0] + 1LL);
LABEL_44:
  v34 = *(__m128i **)(v7 + 32);
  if ( *(_QWORD *)(v7 + 24) - (_QWORD)v34 <= 0x19u )
  {
    result = sub_CB6200(v7, "\"]; // defined externally\n", 26);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F55C50);
    qmemcpy(&v34[1], "xternally\n", 10);
    result = 2681;
    *v34 = si128;
    *(_QWORD *)(v7 + 32) += 26LL;
  }
  if ( v37 != v39 )
    return j_j___libc_free_0(v37, v39[0] + 1LL);
  return result;
}
