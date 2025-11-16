// Function: sub_32664A0
// Address: 0x32664a0
//
__int64 __fastcall sub_32664A0(const __m128i *a1)
{
  __int64 m128i_i64; // r15
  __int64 v2; // rax
  __int64 v3; // rdx
  unsigned __int64 v4; // rax
  char v5; // r14
  unsigned __int64 v6; // r12
  unsigned __int16 *v7; // rdx
  unsigned __int16 v8; // ax
  __int64 v9; // rdx
  __int64 v10; // rdx
  unsigned __int64 v11; // rdx
  char v12; // al
  unsigned __int64 v13; // rax
  __m128i v14; // xmm0
  __m128i v15; // xmm1
  char v16; // r14
  unsigned __int64 v17; // rbx
  unsigned __int16 *v18; // rdx
  unsigned __int16 v19; // ax
  __int64 v20; // rdx
  __int64 v21; // r14
  __int64 v22; // r14
  unsigned int v23; // r12d
  __m128i v25; // xmm5
  __int64 v26; // rax
  __int64 v27; // r12
  __int64 v28; // r12
  unsigned int v29; // ebx
  unsigned __int16 v30; // [rsp+10h] [rbp-90h] BYREF
  __int64 v31; // [rsp+18h] [rbp-88h]
  __int64 v32; // [rsp+20h] [rbp-80h]
  __int64 v33; // [rsp+28h] [rbp-78h]
  __int64 v34; // [rsp+30h] [rbp-70h]
  __int64 v35; // [rsp+38h] [rbp-68h]
  unsigned __int64 v36; // [rsp+40h] [rbp-60h] BYREF
  __int64 v37; // [rsp+48h] [rbp-58h]
  __m128i v38; // [rsp+50h] [rbp-50h] BYREF
  __m128i v39; // [rsp+60h] [rbp-40h] BYREF

  m128i_i64 = (__int64)a1[-2].m128i_i64;
  v38 = _mm_loadu_si128(a1);
  v39 = _mm_loadu_si128(a1 + 1);
  while ( 1 )
  {
    v16 = *(_BYTE *)sub_2E79000(*(__int64 **)(v39.m128i_i64[1] + 40));
    v17 = (unsigned __int32)v39.m128i_i32[0] >> 3;
    v18 = *(unsigned __int16 **)(v38.m128i_i64[1] + 48);
    v19 = *v18;
    v20 = *((_QWORD *)v18 + 1);
    v30 = v19;
    v31 = v20;
    if ( v19 )
    {
      if ( v19 == 1 || (unsigned __int16)(v19 - 504) <= 7u )
LABEL_27:
        BUG();
      v3 = 16LL * (v19 - 1);
      v2 = *(_QWORD *)&byte_444C4A0[v3];
      LOBYTE(v3) = byte_444C4A0[v3 + 8];
    }
    else
    {
      v2 = sub_3007260((__int64)&v30);
      v32 = v2;
      v33 = v3;
    }
    v36 = v2;
    LOBYTE(v37) = v3;
    v4 = sub_CA1930(&v36);
    if ( v16 )
    {
      v27 = (unsigned int)(v4 >> 3);
      sub_3266230((__int64)&v36, (__int64)&v38);
      v28 = v27 - v17;
      if ( (unsigned int)v37 > 0x40 )
      {
        v29 = sub_C44630((__int64)&v36);
        if ( v36 )
          j_j___libc_free_0_0(v36);
      }
      else
      {
        v29 = sub_39FAC40(v36);
      }
      v17 = v28 - (v29 >> 3);
    }
    v5 = *(_BYTE *)sub_2E79000(*(__int64 **)(*(_QWORD *)(m128i_i64 + 24) + 40LL));
    v6 = *(_DWORD *)(m128i_i64 + 16) >> 3;
    v7 = *(unsigned __int16 **)(*(_QWORD *)(m128i_i64 + 8) + 48LL);
    v8 = *v7;
    v9 = *((_QWORD *)v7 + 1);
    LOWORD(v36) = v8;
    v37 = v9;
    if ( v8 )
    {
      if ( v8 == 1 || (unsigned __int16)(v8 - 504) <= 7u )
        goto LABEL_27;
      v26 = 16LL * (v8 - 1);
      v11 = *(_QWORD *)&byte_444C4A0[v26];
      v12 = byte_444C4A0[v26 + 8];
    }
    else
    {
      v34 = sub_3007260((__int64)&v36);
      v35 = v10;
      v11 = v34;
      v12 = v35;
    }
    v36 = v11;
    LOBYTE(v37) = v12;
    v13 = sub_CA1930(&v36);
    if ( v5 )
      break;
    if ( v6 <= v17 )
      goto LABEL_16;
LABEL_8:
    v14 = _mm_loadu_si128((const __m128i *)m128i_i64);
    v15 = _mm_loadu_si128((const __m128i *)(m128i_i64 + 16));
    m128i_i64 -= 32;
    *(__m128i *)(m128i_i64 + 64) = v14;
    *(__m128i *)(m128i_i64 + 80) = v15;
  }
  v21 = (unsigned int)(v13 >> 3);
  sub_3266230((__int64)&v36, m128i_i64);
  v22 = v21 - v6;
  if ( (unsigned int)v37 > 0x40 )
  {
    v23 = sub_C44630((__int64)&v36);
    if ( v36 )
      j_j___libc_free_0_0(v36);
  }
  else
  {
    v23 = sub_39FAC40(v36);
  }
  if ( v22 - (unsigned __int64)(v23 >> 3) > v17 )
    goto LABEL_8;
LABEL_16:
  v25 = _mm_loadu_si128(&v39);
  *(__m128i *)(m128i_i64 + 32) = _mm_loadu_si128(&v38);
  *(__m128i *)(m128i_i64 + 48) = v25;
  return m128i_i64 + 32;
}
