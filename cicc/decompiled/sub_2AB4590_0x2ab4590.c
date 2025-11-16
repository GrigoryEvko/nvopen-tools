// Function: sub_2AB4590
// Address: 0x2ab4590
//
__int64 *__fastcall sub_2AB4590(__int64 a1, __int64 a2)
{
  unsigned int v3; // eax
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 *v6; // rcx
  __int64 *result; // rax
  __m128i *v8; // rdi
  unsigned __int64 v9; // xmm0_8
  __m128i v10; // xmm2
  __m128i v11; // xmm3
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // r12
  __int64 v14; // r15
  unsigned __int64 v15; // r13
  __int64 m128i_i64; // rdx
  __int64 v17; // rcx
  unsigned __int64 v18; // rsi
  char v19; // al
  __int64 v20; // rsi
  __int64 v21; // rax
  int v22; // r8d
  __int64 v23; // rcx
  __int64 v24; // r9
  __m128i *v25; // rax
  __int64 v26; // r13
  __int64 v27; // r13
  __int64 v28; // r15
  int v29; // r10d
  __int64 v30; // r10
  int v31; // r11d
  unsigned int v32; // r15d
  __int64 *v33; // [rsp+18h] [rbp-158h]
  __int64 v34; // [rsp+28h] [rbp-148h]
  __int64 v35; // [rsp+30h] [rbp-140h]
  __int64 *v36; // [rsp+38h] [rbp-138h]
  unsigned __int64 v37; // [rsp+40h] [rbp-130h]
  unsigned __int64 v38; // [rsp+50h] [rbp-120h]
  _BYTE v39[16]; // [rsp+60h] [rbp-110h] BYREF
  void (__fastcall *v40)(_BYTE *, _BYTE *, __int64); // [rsp+70h] [rbp-100h]
  unsigned __int8 (__fastcall *v41)(_BYTE *, unsigned __int64); // [rsp+78h] [rbp-F8h]
  __m128i v42; // [rsp+80h] [rbp-F0h]
  __m128i v43; // [rsp+90h] [rbp-E0h]
  _BYTE v44[16]; // [rsp+A0h] [rbp-D0h] BYREF
  void (__fastcall *v45)(_BYTE *, _BYTE *, __int64); // [rsp+B0h] [rbp-C0h]
  __int64 v46; // [rsp+B8h] [rbp-B8h]
  __m128i v47; // [rsp+C0h] [rbp-B0h] BYREF
  __m128i v48; // [rsp+D0h] [rbp-A0h] BYREF
  _BYTE v49[16]; // [rsp+E0h] [rbp-90h] BYREF
  void (__fastcall *v50)(_BYTE *, _BYTE *, __int64); // [rsp+F0h] [rbp-80h]
  unsigned __int8 (__fastcall *v51)(_BYTE *, unsigned __int64); // [rsp+F8h] [rbp-78h]
  __m128i v52; // [rsp+100h] [rbp-70h] BYREF
  __m128i v53; // [rsp+110h] [rbp-60h] BYREF
  _BYTE v54[16]; // [rsp+120h] [rbp-50h] BYREF
  void (__fastcall *v55)(_BYTE *, _BYTE *, __int64); // [rsp+130h] [rbp-40h]
  __int64 v56; // [rsp+138h] [rbp-38h]

  ++*(_QWORD *)(a1 + 832);
  v34 = a1 + 832;
  if ( *(_BYTE *)(a1 + 860) )
    goto LABEL_6;
  v3 = 4 * (*(_DWORD *)(a1 + 852) - *(_DWORD *)(a1 + 856));
  v4 = *(unsigned int *)(a1 + 848);
  if ( v3 < 0x20 )
    v3 = 32;
  if ( (unsigned int)v4 <= v3 )
  {
    memset(*(void **)(a1 + 840), -1, 8 * v4);
LABEL_6:
    *(_QWORD *)(a1 + 852) = 0;
    goto LABEL_7;
  }
  sub_C8C990(v34, a2);
LABEL_7:
  v5 = *(_QWORD *)(a1 + 416);
  v6 = *(__int64 **)(v5 + 40);
  result = *(__int64 **)(v5 + 32);
  v35 = a1 + 512;
  v33 = v6;
  v36 = result;
  if ( v6 == result )
    return result;
  do
  {
    v8 = &v47;
    sub_AA72C0(&v47, *v36, 1);
    v9 = _mm_loadu_si128(&v47).m128i_u64[0];
    v40 = 0;
    v37 = v9;
    v38 = _mm_loadu_si128(&v48).m128i_u64[0];
    if ( v50 )
    {
      v8 = (__m128i *)v39;
      v50(v39, v49, 2);
      v41 = v51;
      v40 = v50;
    }
    v10 = _mm_loadu_si128(&v52);
    v11 = _mm_loadu_si128(&v53);
    v45 = 0;
    v42 = v10;
    v43 = v11;
    if ( v55 )
    {
      v8 = (__m128i *)v44;
      v55(v44, v54, 2);
      v46 = v56;
      v45 = v55;
    }
    while ( 1 )
    {
      v12 = v37;
      v13 = v37;
      if ( v37 == v42.m128i_i64[0] )
        break;
      while ( 1 )
      {
        if ( !v13 )
          BUG();
        v14 = *(_QWORD *)(v13 - 16);
        v15 = v13 - 24;
        if ( !*(_BYTE *)(a1 + 540) )
        {
          v8 = (__m128i *)v35;
          if ( sub_C8CA60(v35, v13 - 24) )
            goto LABEL_43;
LABEL_38:
          v19 = *(_BYTE *)(v13 - 24);
          if ( v19 != 61 )
          {
            if ( v19 != 84 )
            {
              if ( v19 == 62 )
                goto LABEL_41;
              goto LABEL_43;
            }
            v20 = *(_QWORD *)(a1 + 440);
            v21 = *(unsigned int *)(v20 + 104);
            v8 = *(__m128i **)(v20 + 88);
            if ( !(_DWORD)v21 )
              goto LABEL_43;
            v22 = v21 - 1;
            LODWORD(v23) = (v21 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
            m128i_i64 = (__int64)v8[(unsigned int)v23].m128i_i64;
            v24 = *(_QWORD *)m128i_i64;
            if ( v15 != *(_QWORD *)m128i_i64 )
            {
              v30 = *(_QWORD *)m128i_i64;
              v31 = (v21 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
              m128i_i64 = 1;
              while ( v30 != -4096 )
              {
                v32 = m128i_i64 + 1;
                v31 = v22 & (m128i_i64 + v31);
                m128i_i64 = (__int64)v8[v31].m128i_i64;
                v30 = *(_QWORD *)m128i_i64;
                if ( v15 == *(_QWORD *)m128i_i64 )
                {
                  v29 = 1;
                  v25 = &v8[v21];
                  if ( (__m128i *)m128i_i64 == v25 )
                    goto LABEL_43;
                  while ( v24 != -4096 )
                  {
                    v23 = v22 & (unsigned int)(v23 + v29);
                    m128i_i64 = (__int64)v8[v23].m128i_i64;
                    v24 = *(_QWORD *)m128i_i64;
                    if ( v15 == *(_QWORD *)m128i_i64 )
                      goto LABEL_47;
                    ++v29;
                  }
                  v26 = *(_QWORD *)(v20 + 112);
                  goto LABEL_59;
                }
                m128i_i64 = v32;
              }
              goto LABEL_43;
            }
            v25 = &v8[v21];
            if ( (__m128i *)m128i_i64 == v25 )
            {
LABEL_43:
              v12 = v37;
              goto LABEL_19;
            }
LABEL_47:
            v26 = *(_QWORD *)(v20 + 112);
            if ( v25 == (__m128i *)m128i_i64 )
LABEL_59:
              v27 = 184LL * *(unsigned int *)(v20 + 120) + v26;
            else
              v27 = 184LL * *(unsigned int *)(m128i_i64 + 8) + v26;
            if ( byte_500D668 )
              goto LABEL_43;
            v8 = *(__m128i **)(a1 + 496);
            if ( !(unsigned __int8)sub_31A4BE0(v8) )
            {
              if ( *(_BYTE *)(v27 + 73) )
                goto LABEL_43;
            }
            v28 = *(_QWORD *)(a1 + 448);
            sub_1022EF0(*(_DWORD *)(v27 + 48));
            v8 = (__m128i *)v28;
            if ( (unsigned __int8)sub_DFE340(v28) )
              goto LABEL_43;
            v14 = *(_QWORD *)(v27 + 64);
            if ( *(_BYTE *)(v13 - 24) == 62 )
LABEL_41:
              v14 = *(_QWORD *)(*(_QWORD *)(v13 - 88) + 8LL);
          }
          v8 = (__m128i *)v34;
          sub_AE6EC0(v34, v14);
          goto LABEL_43;
        }
        m128i_i64 = *(_QWORD *)(a1 + 520);
        v17 = m128i_i64 + 8LL * *(unsigned int *)(a1 + 532);
        if ( m128i_i64 == v17 )
          goto LABEL_38;
        while ( v15 != *(_QWORD *)m128i_i64 )
        {
          m128i_i64 += 8;
          if ( v17 == m128i_i64 )
            goto LABEL_38;
        }
LABEL_19:
        v12 = *(_QWORD *)(v12 + 8);
        v37 = v12;
        v13 = v12;
        v18 = v12;
        if ( v12 != v38 )
          break;
LABEL_26:
        if ( v42.m128i_i64[0] == v13 )
          goto LABEL_27;
      }
      while ( 1 )
      {
        if ( v18 )
          v18 -= 24LL;
        if ( !v40 )
          sub_4263D6(v8, v18, m128i_i64);
        v8 = (__m128i *)v39;
        if ( v41(v39, v18) )
          break;
        m128i_i64 = 0;
        v18 = *(_QWORD *)(v37 + 8);
        v37 = v18;
        v12 = v18;
        if ( v38 == v18 )
        {
          v13 = v18;
          goto LABEL_26;
        }
      }
    }
LABEL_27:
    if ( v45 )
      v45(v44, v44, 3);
    if ( v40 )
      v40(v39, v39, 3);
    if ( v55 )
      v55(v54, v54, 3);
    if ( v50 )
      v50(v49, v49, 3);
    result = ++v36;
  }
  while ( v33 != v36 );
  return result;
}
