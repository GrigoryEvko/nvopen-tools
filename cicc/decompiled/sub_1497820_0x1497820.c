// Function: sub_1497820
// Address: 0x1497820
//
__int64 __fastcall sub_1497820(__m128i *a1, const __m128i *a2)
{
  __int64 v4; // rax
  __int64 m128i_i64; // r13
  unsigned __int32 v6; // eax
  __m128i v7; // xmm0
  __int32 v8; // eax
  __int64 v9; // rax
  __int64 v10; // rbx
  __int64 result; // rax
  __int64 v12; // r15
  void *v13; // rax
  const void *v14; // rsi
  __int64 v15; // rax
  __int64 v16; // rdx
  __int32 v17; // edi
  __int64 v18; // r12
  unsigned __int64 v19; // r15
  _QWORD *v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rdx
  _QWORD *v23; // rax
  __int64 v24; // rdx
  _QWORD *v25; // rdx
  char v26; // cl
  __int64 v27; // rax
  __int64 v28; // r12
  int v29; // eax
  void *v30; // [rsp+10h] [rbp-90h] BYREF
  _QWORD v31[5]; // [rsp+18h] [rbp-88h] BYREF
  void *v32; // [rsp+40h] [rbp-60h] BYREF
  __int64 v33; // [rsp+48h] [rbp-58h] BYREF
  __int64 v34; // [rsp+50h] [rbp-50h]
  __int64 v35; // [rsp+58h] [rbp-48h]
  __int64 i; // [rsp+60h] [rbp-40h]

  a1->m128i_i64[0] = 0;
  a1->m128i_i64[1] = 0;
  a1[1].m128i_i64[0] = 0;
  a1[1].m128i_i32[2] = 0;
  j___libc_free_0(0);
  v4 = a2[1].m128i_u32[2];
  a1[1].m128i_i32[2] = v4;
  if ( (_DWORD)v4 )
  {
    v13 = (void *)sub_22077B0(24 * v4);
    v14 = (const void *)a2->m128i_i64[1];
    a1->m128i_i64[1] = (__int64)v13;
    a1[1].m128i_i64[0] = a2[1].m128i_i64[0];
    memcpy(v13, v14, 24LL * a1[1].m128i_u32[2]);
  }
  else
  {
    a1->m128i_i64[1] = 0;
    a1[1].m128i_i64[0] = 0;
  }
  a1[2].m128i_i64[0] = 0;
  m128i_i64 = (__int64)a1[2].m128i_i64;
  v6 = sub_1454B60(0x56u);
  a1[3].m128i_i32[2] = v6;
  if ( v6 )
  {
    v23 = (_QWORD *)sub_22077B0(48LL * v6);
    a1[3].m128i_i64[0] = 0;
    a1[2].m128i_i64[1] = (__int64)v23;
    v24 = a1[3].m128i_u32[2];
    v33 = 2;
    v34 = 0;
    v35 = -8;
    v32 = &unk_49EC740;
    v25 = &v23[6 * v24];
    for ( i = 0; v25 != v23; v23 += 6 )
    {
      if ( v23 )
      {
        v26 = v33;
        v23[2] = 0;
        v23[3] = -8;
        *v23 = &unk_49EC740;
        v23[1] = v26 & 6;
        v23[4] = i;
      }
    }
  }
  else
  {
    a1[2].m128i_i64[1] = 0;
    a1[3].m128i_i64[0] = 0;
  }
  a1[6].m128i_i8[0] = 0;
  a1[6].m128i_i8[9] = 1;
  a1[7].m128i_i64[0] = a2[7].m128i_i64[0];
  a1[7].m128i_i64[1] = a2[7].m128i_i64[1];
  v7 = _mm_loadu_si128(a2 + 9);
  a1[8].m128i_i64[1] = a2[8].m128i_i64[1];
  v8 = a2[10].m128i_i32[0];
  a1[9] = v7;
  a1[10].m128i_i32[0] = v8;
  a1[8].m128i_i64[0] = (__int64)&unk_49EC708;
  a1[10].m128i_i64[1] = (__int64)&a1[11].m128i_i64[1];
  a1[11].m128i_i64[0] = 0x1000000000LL;
  if ( a2[11].m128i_i32[0] )
    sub_14531E0((__int64)&a1[10].m128i_i64[1], (__int64)&a2[10].m128i_i64[1]);
  a1[19].m128i_i64[1] = 0;
  a1[20].m128i_i64[0] = 0;
  a1[20].m128i_i64[1] = 0;
  a1[21].m128i_i32[0] = 0;
  j___libc_free_0(0);
  v9 = a2[21].m128i_u32[0];
  a1[21].m128i_i32[0] = v9;
  if ( (_DWORD)v9 )
  {
    v15 = sub_22077B0(56 * v9);
    v16 = a2[20].m128i_i64[1];
    v17 = a1[21].m128i_i32[0];
    a1[20].m128i_i64[0] = v15;
    a1[20].m128i_i64[1] = v16;
    if ( v17 )
    {
      v18 = 0;
      v19 = 0;
      while ( 1 )
      {
        v20 = (_QWORD *)(v18 + v15);
        if ( v20 )
        {
          *v20 = *(_QWORD *)(a2[20].m128i_i64[0] + v18);
          v20 = (_QWORD *)(v18 + a1[20].m128i_i64[0]);
        }
        if ( *v20 != -16 && *v20 != -8 )
        {
          v21 = a2[20].m128i_i64[0];
          v20[2] = 0x400000000LL;
          v20[1] = v20 + 3;
          v22 = v18 + v21;
          if ( *(_DWORD *)(v22 + 16) )
            sub_14531E0((__int64)(v20 + 1), v22 + 8);
        }
        ++v19;
        v18 += 56;
        if ( a1[21].m128i_u32[0] <= v19 )
          break;
        v15 = a1[20].m128i_i64[0];
      }
    }
  }
  else
  {
    a1[20].m128i_i64[0] = 0;
    a1[20].m128i_i64[1] = 0;
  }
  a1[21].m128i_i32[2] = a2[21].m128i_i32[2];
  a1[22].m128i_i64[0] = a2[22].m128i_i64[0];
  v10 = a2[2].m128i_i64[1];
  result = a2[3].m128i_u32[0];
  v12 = v10 + 48LL * a2[3].m128i_u32[2];
  if ( (_DWORD)result )
  {
    v31[1] = 0;
    v31[0] = 2;
    v31[2] = -8;
    v30 = &unk_49EC740;
    v31[3] = 0;
    v33 = 2;
    v34 = 0;
    v35 = -16;
    for ( i = 0; v10 != v12; v10 += 48 )
    {
      v27 = *(_QWORD *)(v10 + 24);
      if ( v27 != -8 && v27 != -16 )
        break;
    }
    v32 = &unk_49EE2B0;
    sub_1455FA0((__int64)&v33);
    v30 = &unk_49EE2B0;
    sub_1455FA0((__int64)v31);
    result = a2[3].m128i_u32[2];
    v28 = a2[2].m128i_i64[1] + 48 * result;
    while ( v10 != v28 )
    {
      v29 = *(_DWORD *)(v10 + 40);
      v30 = *(void **)(v10 + 24);
      LODWORD(v31[0]) = v29;
      result = sub_14974D0((__int64)&v32, m128i_i64, (__int64)&v30);
      do
      {
        v10 += 48;
        if ( v10 == v12 )
          break;
        result = *(_QWORD *)(v10 + 24);
      }
      while ( result == -8 || result == -16 );
    }
  }
  return result;
}
