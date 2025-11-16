// Function: sub_3592050
// Address: 0x3592050
//
__int64 __fastcall sub_3592050(__int64 a1, __int64 a2)
{
  unsigned int v4; // r15d
  unsigned int v5; // r14d
  unsigned __int64 v6; // rdx
  unsigned int v7; // ecx
  unsigned __int64 v8; // rsi
  unsigned int v9; // r8d
  unsigned __int64 v10; // rsi
  __m128i *v11; // rsi
  int v12; // ecx
  unsigned int v13; // eax
  unsigned int v14; // r9d
  __int64 v15; // rax
  __int64 v16; // r10
  __int64 v17; // rdx
  __m128i *v18; // rax
  unsigned __int64 *v19; // rax
  size_t v20; // rsi
  __int64 v21; // r15
  __int64 i; // rbx
  int v23; // eax
  __int64 v24; // rax
  int v25; // eax
  __int64 v26; // rax
  __int64 v27; // r10
  unsigned int v28; // r13d
  const __m128i *v29; // rbx
  unsigned __int64 v30; // r12
  unsigned __int64 v31; // rdi
  unsigned __int64 *v33; // rax
  const __m128i *v34; // rsi
  _OWORD *v35; // rdi
  __int64 v36; // r15
  __int64 v37; // [rsp+8h] [rbp-F8h]
  const __m128i *v38; // [rsp+20h] [rbp-E0h] BYREF
  const __m128i *v39; // [rsp+28h] [rbp-D8h]
  const __m128i *v40; // [rsp+30h] [rbp-D0h]
  __m128i *v41; // [rsp+40h] [rbp-C0h]
  size_t v42; // [rsp+48h] [rbp-B8h]
  __m128i v43; // [rsp+50h] [rbp-B0h] BYREF
  __int64 v44[2]; // [rsp+60h] [rbp-A0h] BYREF
  __int64 v45; // [rsp+70h] [rbp-90h] BYREF
  __m128i *v46; // [rsp+80h] [rbp-80h] BYREF
  unsigned __int64 v47; // [rsp+88h] [rbp-78h]
  __m128i v48; // [rsp+90h] [rbp-70h] BYREF
  _OWORD *v49; // [rsp+A0h] [rbp-60h] BYREF
  _OWORD *v50; // [rsp+A8h] [rbp-58h] BYREF
  _OWORD v51[5]; // [rsp+B0h] [rbp-50h] BYREF

  v4 = *(_DWORD *)(a1 + 8);
  v38 = 0;
  v39 = 0;
  v40 = 0;
  if ( v4 <= 9 )
  {
    v46 = &v48;
    sub_2240A50((__int64 *)&v46, 1u, 0);
    v11 = v46;
LABEL_16:
    v11->m128i_i8[0] = v4 + 48;
    goto LABEL_17;
  }
  if ( v4 <= 0x63 )
  {
    v46 = &v48;
    sub_2240A50((__int64 *)&v46, 2u, 0);
    v11 = v46;
  }
  else
  {
    if ( v4 <= 0x3E7 )
    {
      v10 = 3;
      v5 = v4;
    }
    else
    {
      v5 = v4;
      v6 = v4;
      if ( v4 <= 0x270F )
      {
        v10 = 4;
      }
      else
      {
        v7 = 1;
        do
        {
          v8 = v6;
          v9 = v7;
          v7 += 4;
          v6 /= 0x2710u;
          if ( v8 <= 0x1869F )
          {
            v10 = v7;
            goto LABEL_11;
          }
          if ( (unsigned int)v6 <= 0x63 )
          {
            v46 = &v48;
            v10 = v9 + 5;
            goto LABEL_12;
          }
          if ( (unsigned int)v6 <= 0x3E7 )
          {
            v10 = v9 + 6;
            goto LABEL_11;
          }
        }
        while ( (unsigned int)v6 > 0x270F );
        v10 = v9 + 7;
      }
    }
LABEL_11:
    v46 = &v48;
LABEL_12:
    sub_2240A50((__int64 *)&v46, v10, 0);
    v11 = v46;
    v12 = v47 - 1;
    while ( 1 )
    {
      v13 = v4 - 100 * (v5 / 0x64);
      v14 = v4;
      v4 = v5 / 0x64;
      v15 = 2 * v13;
      v16 = (unsigned int)(v15 + 1);
      LOBYTE(v15) = a00010203040506[v15];
      v11->m128i_i8[v12] = a00010203040506[v16];
      v17 = (unsigned int)(v12 - 1);
      v12 -= 2;
      v11->m128i_i8[v17] = v15;
      if ( v14 <= 0x270F )
        break;
      v5 /= 0x64u;
    }
    if ( v14 <= 0x3E7 )
      goto LABEL_16;
  }
  v36 = 2 * v4;
  v11->m128i_i8[1] = a00010203040506[(unsigned int)(v36 + 1)];
  v11->m128i_i8[0] = a00010203040506[v36];
LABEL_17:
  v18 = (__m128i *)sub_2241130((unsigned __int64 *)&v46, 0, 0, "bb", 2u);
  v49 = v51;
  if ( (__m128i *)v18->m128i_i64[0] == &v18[1] )
  {
    v51[0] = _mm_loadu_si128(v18 + 1);
  }
  else
  {
    v49 = (_OWORD *)v18->m128i_i64[0];
    *(_QWORD *)&v51[0] = v18[1].m128i_i64[0];
  }
  v50 = (_OWORD *)v18->m128i_i64[1];
  v18->m128i_i64[0] = (__int64)v18[1].m128i_i64;
  v18->m128i_i64[1] = 0;
  v18[1].m128i_i8[0] = 0;
  if ( v50 == (_OWORD *)0x3FFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"basic_string::append");
  v19 = sub_2241490((unsigned __int64 *)&v49, "_", 1u);
  v41 = &v43;
  if ( (unsigned __int64 *)*v19 == v19 + 2 )
  {
    v43 = _mm_loadu_si128((const __m128i *)v19 + 1);
  }
  else
  {
    v41 = (__m128i *)*v19;
    v43.m128i_i64[0] = v19[2];
  }
  v20 = v19[1];
  *((_BYTE *)v19 + 16) = 0;
  v42 = v20;
  *v19 = (unsigned __int64)(v19 + 2);
  v19[1] = 0;
  if ( v49 != v51 )
    j_j___libc_free_0((unsigned __int64)v49);
  if ( v46 != &v48 )
    j_j___libc_free_0((unsigned __int64)v46);
  v21 = *(_QWORD *)(a2 + 56);
  for ( i = a2 + 48; i != v21; v21 = *(_QWORD *)(v21 + 8) )
  {
    while ( 1 )
    {
      if ( (unsigned int)*(unsigned __int16 *)(v21 + 68) - 1 > 1
        || (*(_BYTE *)(*(_QWORD *)(v21 + 32) + 64LL) & 0x10) == 0 )
      {
        v23 = *(_DWORD *)(v21 + 44);
        if ( (v23 & 4) != 0 || (v23 & 8) == 0 )
          v24 = (*(_QWORD *)(*(_QWORD *)(v21 + 16) + 24LL) >> 20) & 1LL;
        else
          LOBYTE(v24) = sub_2E88A90(v21, 0x100000, 1);
        if ( !(_BYTE)v24 )
        {
          v25 = *(_DWORD *)(v21 + 44);
          if ( (v25 & 4) != 0 || (v25 & 8) == 0 )
            v26 = (*(_QWORD *)(*(_QWORD *)(v21 + 16) + 24LL) >> 10) & 1LL;
          else
            LOBYTE(v26) = sub_2E88A90(v21, 1024, 1);
          if ( !(_BYTE)v26 && (*(_DWORD *)(v21 + 40) & 0xFFFFFF) != 0 )
          {
            v27 = *(_QWORD *)(v21 + 32);
            if ( !*(_BYTE *)v27 )
            {
              v37 = *(_QWORD *)(v21 + 32);
              if ( *(int *)(v27 + 8) < 0 )
              {
                sub_3590C00(v44, (__int64 *)a1, v21);
                v33 = sub_2241130((unsigned __int64 *)v44, 0, 0, v41, v42);
                v46 = &v48;
                if ( (unsigned __int64 *)*v33 == v33 + 2 )
                {
                  v48 = _mm_loadu_si128((const __m128i *)v33 + 1);
                }
                else
                {
                  v46 = (__m128i *)*v33;
                  v48.m128i_i64[0] = v33[2];
                }
                v47 = v33[1];
                *v33 = (unsigned __int64)(v33 + 2);
                v33[1] = 0;
                *((_BYTE *)v33 + 16) = 0;
                LODWORD(v49) = *(_DWORD *)(v37 + 8);
                v50 = (_OWORD *)((char *)v51 + 8);
                sub_35907E0((__int64 *)&v50, v46, (__int64)v46->m128i_i64 + v47);
                v34 = v39;
                if ( v39 == v40 )
                {
                  sub_35914E0((unsigned __int64 *)&v38, v39, (__int64)&v49);
                  v35 = v50;
                }
                else
                {
                  v35 = v50;
                  if ( v39 )
                  {
                    v39->m128i_i32[0] = (int)v49;
                    v34->m128i_i64[1] = (__int64)&v34[1].m128i_i64[1];
                    if ( v50 == (_OWORD *)((char *)v51 + 8) )
                    {
                      *(const __m128i *)((char *)&v34[1] + 8) = _mm_loadu_si128((const __m128i *)((char *)v51 + 8));
                    }
                    else
                    {
                      v34->m128i_i64[1] = (__int64)v50;
                      v34[1].m128i_i64[1] = *((_QWORD *)&v51[0] + 1);
                    }
                    v34[1].m128i_i64[0] = *(_QWORD *)&v51[0];
                    v34 = v39;
                    *(_QWORD *)&v51[0] = 0;
                    BYTE8(v51[0]) = 0;
                    v50 = (_OWORD *)((char *)v51 + 8);
                    v35 = (_OWORD *)((char *)v51 + 8);
                  }
                  v39 = (const __m128i *)((char *)v34 + 40);
                }
                if ( v35 != (_OWORD *)((char *)v51 + 8) )
                  j_j___libc_free_0((unsigned __int64)v35);
                if ( v46 != &v48 )
                  j_j___libc_free_0((unsigned __int64)v46);
                if ( (__int64 *)v44[0] != &v45 )
                  j_j___libc_free_0(v44[0]);
              }
            }
          }
        }
      }
      if ( (*(_BYTE *)v21 & 4) == 0 )
        break;
      v21 = *(_QWORD *)(v21 + 8);
      if ( i == v21 )
        goto LABEL_44;
    }
    while ( (*(_BYTE *)(v21 + 44) & 8) != 0 )
      v21 = *(_QWORD *)(v21 + 8);
  }
LABEL_44:
  v28 = 0;
  if ( v39 != v38 )
  {
    sub_3591920(&v49, (__int64 *)a1, (__int64 *)&v38);
    v28 = sub_3590A60((_QWORD **)a1, (__int64)&v49);
    sub_3590890(*(unsigned __int64 *)&v51[0]);
  }
  if ( v41 != &v43 )
    j_j___libc_free_0((unsigned __int64)v41);
  v29 = v39;
  v30 = (unsigned __int64)v38;
  if ( v39 != v38 )
  {
    do
    {
      v31 = *(_QWORD *)(v30 + 8);
      if ( v31 != v30 + 24 )
        j_j___libc_free_0(v31);
      v30 += 40LL;
    }
    while ( v29 != (const __m128i *)v30 );
    v30 = (unsigned __int64)v38;
  }
  if ( v30 )
    j_j___libc_free_0(v30);
  return v28;
}
