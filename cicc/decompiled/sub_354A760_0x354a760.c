// Function: sub_354A760
// Address: 0x354a760
//
void __fastcall sub_354A760(__int64 a1)
{
  __m128i *v2; // r12
  __int64 v3; // rax
  __int64 v4; // rbx
  __int64 v5; // rax
  __int64 v6; // r14
  __int64 i; // rbx
  __int64 v8; // rax
  __int64 *v9; // r14
  __int64 v10; // rbx
  unsigned __int64 v11; // rax
  __m128i *v12; // r11
  __m128i *j; // rdi
  unsigned __int64 v14; // rsi
  unsigned __int64 v15; // rdx
  __m128i *v16; // rcx
  __m128i *v17; // rax
  __int32 v18; // r8d
  __int32 v19; // edx
  __int64 v20; // r12
  __int64 v21; // r14
  __int64 v22; // rax
  __int64 v23; // r11
  __int64 *v24; // rdi
  __int64 v25; // r15
  __int64 v26; // r9
  unsigned int v27; // esi
  __int64 v28; // r10
  __int64 v29; // r9
  __int64 v30; // rax
  __int64 v31; // r9
  __int64 v32; // r11
  unsigned int v33; // ecx
  __int64 v34; // r10
  __int64 v35; // [rsp+8h] [rbp-68h]
  __m128i v36; // [rsp+10h] [rbp-60h] BYREF
  __m128i *v37; // [rsp+20h] [rbp-50h] BYREF
  __m128i *v38; // [rsp+28h] [rbp-48h]
  __m128i *v39; // [rsp+30h] [rbp-40h]

  v2 = 0;
  v3 = *(unsigned int *)(a1 + 4008);
  v37 = 0;
  v38 = 0;
  v39 = 0;
  v4 = 16 * v3;
  if ( v3 )
  {
    v5 = sub_22077B0(16 * v3);
    v2 = (__m128i *)(v5 + v4);
    v37 = (__m128i *)v5;
    v39 = (__m128i *)(v5 + v4);
    do
    {
      if ( v5 )
      {
        *(_QWORD *)v5 = 0;
        *(_DWORD *)(v5 + 8) = 0;
      }
      v5 += 16;
    }
    while ( (__m128i *)v5 != v2 );
    v6 = *(unsigned int *)(a1 + 4008);
    v38 = v2;
    if ( (_DWORD)v6 )
    {
      for ( i = 0; i != v6; ++i )
      {
        while ( 1 )
        {
          v8 = *(_QWORD *)(*(_QWORD *)(a1 + 4000) + 8 * i);
          v36.m128i_i32[2] = i;
          v36.m128i_i64[0] = v8;
          if ( v2 != v39 )
            break;
          ++i;
          sub_354A5E0((unsigned __int64 *)&v37, v2, &v36);
          v2 = v38;
          if ( v6 == i )
            goto LABEL_13;
        }
        if ( v2 )
        {
          *v2 = _mm_loadu_si128(&v36);
          v2 = v38;
        }
        v38 = ++v2;
      }
    }
  }
LABEL_13:
  v9 = (__int64 *)v37;
  if ( v2 != v37 )
  {
    v10 = (char *)v2 - (char *)v37;
    _BitScanReverse64(&v11, v2 - v37);
    sub_353E330((__int64)v37, v2, 2LL * (int)(63 - (v11 ^ 0x3F)));
    if ( v10 <= 256 )
    {
      sub_353D310((__int64)v9, (unsigned __int64 *)v2);
    }
    else
    {
      sub_353D310((__int64)v9, (unsigned __int64 *)v9 + 32);
      for ( j = v12; j != v2; v16->m128i_i32[2] = v18 )
      {
        v14 = j->m128i_i64[0];
        v15 = j[-1].m128i_u64[0];
        v16 = j;
        v17 = j - 1;
        v18 = j->m128i_i32[2];
        if ( j->m128i_i64[0] < v15 )
        {
          do
          {
            v17[1].m128i_i64[0] = v15;
            v19 = v17->m128i_i32[2];
            v16 = v17--;
            v17[2].m128i_i32[2] = v19;
            v15 = v17->m128i_i64[0];
          }
          while ( v14 < v17->m128i_i64[0] );
        }
        ++j;
        v16->m128i_i64[0] = v14;
      }
    }
  }
  v20 = 0;
  v35 = *(unsigned int *)(a1 + 4008);
  if ( (_DWORD)v35 )
  {
    do
    {
      v21 = *(_QWORD *)(*(_QWORD *)(a1 + 4000) + 8 * v20);
      v22 = sub_35459D0(*(_QWORD **)(a1 + 3464), v21);
      if ( *(_QWORD *)v22 != *(_QWORD *)v22 + 32LL * *(unsigned int *)(v22 + 8) )
      {
        v23 = (__int64)v38;
        v24 = (__int64 *)v37;
        v25 = *(_QWORD *)v22;
        do
        {
          v26 = *(_QWORD *)(v25 + 8);
          v36.m128i_i32[2] = 0;
          v36.m128i_i64[0] = v26 & 0xFFFFFFFFFFFFFFF8LL;
          v27 = *((_DWORD *)sub_353D9C0(v24, v23, (unsigned __int64 *)&v36) + 2);
          if ( *(_WORD *)(*(_QWORD *)v29 + 68LL) != 0
            && *(_WORD *)(*(_QWORD *)v29 + 68LL) != 68
            && v27 < (unsigned int)v20 )
          {
            break;
          }
          v25 += 32;
        }
        while ( v28 != v25 );
      }
      v30 = sub_3545E90(*(_QWORD **)(a1 + 3464), v21);
      v31 = *(_QWORD *)v30;
      v32 = *(_QWORD *)v30 + 32LL * *(unsigned int *)(v30 + 8);
      if ( v32 != *(_QWORD *)v30 )
      {
        do
        {
          if ( *(_DWORD *)(*(_QWORD *)v31 + 200LL) != -1 )
          {
            v36.m128i_i64[0] = *(_QWORD *)v31;
            v36.m128i_i32[2] = 0;
            v33 = *((_DWORD *)sub_353D9C0(v37, (__int64)v38, (unsigned __int64 *)&v36) + 2);
            if ( *(_WORD *)(*(_QWORD *)v34 + 68LL) != 68
              && *(_WORD *)(*(_QWORD *)v34 + 68LL) != 0
              && v33 < (unsigned int)v20 )
            {
              break;
            }
          }
          v31 += 32;
        }
        while ( v32 != v31 );
      }
      ++v20;
    }
    while ( v35 != v20 );
  }
  if ( v37 )
    j_j___libc_free_0((unsigned __int64)v37);
}
