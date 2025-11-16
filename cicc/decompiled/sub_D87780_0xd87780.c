// Function: sub_D87780
// Address: 0xd87780
//
__m128i *__fastcall sub_D87780(__int64 a1, __int64 a2, _QWORD *a3)
{
  __m128i *v5; // r14
  __int64 v6; // rax
  __int64 v7; // rcx
  __int64 m128i_i64; // r8
  unsigned int v9; // eax
  unsigned int v10; // eax
  __int32 v11; // eax
  __int64 v12; // rdi
  __int64 v13; // r12
  __m128i *v14; // r15
  __m128i *v15; // rbx
  const void **v16; // rsi
  __int64 v17; // rax
  __int64 v18; // rdi
  __int64 v19; // rdi
  unsigned int v20; // eax
  unsigned int v21; // eax
  __int32 v22; // eax
  __int64 v23; // rdi
  unsigned int v24; // eax
  unsigned int v25; // eax
  __int64 v27; // rdx
  __int64 j; // rax
  __int64 v29; // rax
  unsigned int v30; // eax
  unsigned int v31; // eax
  __int64 v32; // rdx
  __int64 i; // rax
  __int64 v34; // rax
  const void **v35; // [rsp+8h] [rbp-38h]

  v5 = (__m128i *)a3[1];
  v35 = (const void **)(a1 + 48);
  if ( v5 )
  {
    v6 = v5->m128i_i64[1];
    a3[1] = v6;
    if ( v6 )
    {
      if ( v5 == *(__m128i **)(v6 + 24) )
      {
        *(_QWORD *)(v6 + 24) = 0;
        v32 = *(_QWORD *)(a3[1] + 16LL);
        if ( v32 )
        {
          a3[1] = v32;
          for ( i = *(_QWORD *)(v32 + 24); i; i = *(_QWORD *)(i + 24) )
          {
            a3[1] = i;
            v32 = i;
          }
          v34 = *(_QWORD *)(v32 + 16);
          if ( v34 )
            a3[1] = v34;
        }
      }
      else
      {
        *(_QWORD *)(v6 + 16) = 0;
      }
    }
    else
    {
      *a3 = 0;
    }
    sub_969240(v5[4].m128i_i64);
    sub_969240(v5[3].m128i_i64);
    m128i_i64 = (__int64)v5[4].m128i_i64;
    v5[2] = _mm_loadu_si128((const __m128i *)(a1 + 32));
    v9 = *(_DWORD *)(a1 + 56);
    v5[3].m128i_i32[2] = v9;
    if ( v9 > 0x40 )
    {
      sub_C43780((__int64)v5[3].m128i_i64, v35);
      m128i_i64 = (__int64)v5[4].m128i_i64;
    }
    else
    {
      v5[3].m128i_i64[0] = *(_QWORD *)(a1 + 48);
    }
    v10 = *(_DWORD *)(a1 + 72);
    v5[4].m128i_i32[2] = v10;
    if ( v10 <= 0x40 )
      goto LABEL_8;
    sub_C43780(m128i_i64, (const void **)(a1 + 64));
  }
  else
  {
    v5 = (__m128i *)sub_22077B0(80);
    v5[2] = _mm_loadu_si128((const __m128i *)(a1 + 32));
    v30 = *(_DWORD *)(a1 + 56);
    v5[3].m128i_i32[2] = v30;
    if ( v30 > 0x40 )
      sub_C43780((__int64)v5[3].m128i_i64, v35);
    else
      v5[3].m128i_i64[0] = *(_QWORD *)(a1 + 48);
    v31 = *(_DWORD *)(a1 + 72);
    v5[4].m128i_i32[2] = v31;
    if ( v31 <= 0x40 )
    {
LABEL_8:
      v5[4].m128i_i64[0] = *(_QWORD *)(a1 + 64);
      goto LABEL_9;
    }
    sub_C43780((__int64)v5[4].m128i_i64, (const void **)(a1 + 64));
  }
LABEL_9:
  v11 = *(_DWORD *)a1;
  v5[1].m128i_i64[0] = 0;
  v5[1].m128i_i64[1] = 0;
  v5->m128i_i32[0] = v11;
  v5->m128i_i64[1] = a2;
  v12 = *(_QWORD *)(a1 + 24);
  if ( v12 )
    v5[1].m128i_i64[1] = sub_D87780(v12, v5, a3, v7, m128i_i64);
  v13 = *(_QWORD *)(a1 + 16);
  v14 = v5;
  if ( v13 )
  {
    v15 = (__m128i *)a3[1];
    v16 = (const void **)(v13 + 48);
    if ( v15 )
      goto LABEL_13;
LABEL_29:
    v15 = (__m128i *)sub_22077B0(80);
    v15[2] = _mm_loadu_si128((const __m128i *)(v13 + 32));
    v24 = *(_DWORD *)(v13 + 56);
    v15[3].m128i_i32[2] = v24;
    if ( v24 <= 0x40 )
    {
      while ( 1 )
      {
        v15[3].m128i_i64[0] = *(_QWORD *)(v13 + 48);
        v21 = *(_DWORD *)(v13 + 72);
        v15[4].m128i_i32[2] = v21;
        if ( v21 > 0x40 )
          break;
LABEL_24:
        v15[4].m128i_i64[0] = *(_QWORD *)(v13 + 64);
LABEL_25:
        v22 = *(_DWORD *)v13;
        v15[1].m128i_i64[0] = 0;
        v15[1].m128i_i64[1] = 0;
        v15->m128i_i32[0] = v22;
        v14[1].m128i_i64[0] = (__int64)v15;
        v15->m128i_i64[1] = (__int64)v14;
        v23 = *(_QWORD *)(v13 + 24);
        if ( v23 )
          v15[1].m128i_i64[1] = sub_D87780(v23, v15, a3, v7, m128i_i64);
        v13 = *(_QWORD *)(v13 + 16);
        if ( !v13 )
          return v5;
        v14 = v15;
        v15 = (__m128i *)a3[1];
        v16 = (const void **)(v13 + 48);
        if ( !v15 )
          goto LABEL_29;
LABEL_13:
        v17 = v15->m128i_i64[1];
        a3[1] = v17;
        if ( v17 )
        {
          if ( v15 == *(__m128i **)(v17 + 24) )
          {
            *(_QWORD *)(v17 + 24) = 0;
            v27 = *(_QWORD *)(a3[1] + 16LL);
            if ( v27 )
            {
              a3[1] = v27;
              for ( j = *(_QWORD *)(v27 + 24); j; j = *(_QWORD *)(j + 24) )
              {
                a3[1] = j;
                v27 = j;
              }
              v29 = *(_QWORD *)(v27 + 16);
              if ( v29 )
                a3[1] = v29;
            }
          }
          else
          {
            *(_QWORD *)(v17 + 16) = 0;
          }
        }
        else
        {
          *a3 = 0;
        }
        if ( v15[4].m128i_i32[2] > 0x40u )
        {
          v18 = v15[4].m128i_i64[0];
          if ( v18 )
            j_j___libc_free_0_0(v18);
        }
        if ( v15[3].m128i_i32[2] > 0x40u )
        {
          v19 = v15[3].m128i_i64[0];
          if ( v19 )
            j_j___libc_free_0_0(v19);
        }
        v15[2] = _mm_loadu_si128((const __m128i *)(v13 + 32));
        v20 = *(_DWORD *)(v13 + 56);
        v15[3].m128i_i32[2] = v20;
        if ( v20 > 0x40 )
          goto LABEL_30;
      }
    }
    else
    {
LABEL_30:
      sub_C43780((__int64)v15[3].m128i_i64, v16);
      v25 = *(_DWORD *)(v13 + 72);
      v15[4].m128i_i32[2] = v25;
      if ( v25 <= 0x40 )
        goto LABEL_24;
    }
    sub_C43780((__int64)v15[4].m128i_i64, (const void **)(v13 + 64));
    goto LABEL_25;
  }
  return v5;
}
