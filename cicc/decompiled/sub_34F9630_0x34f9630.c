// Function: sub_34F9630
// Address: 0x34f9630
//
__int64 __fastcall sub_34F9630(__int64 a1, const __m128i *a2)
{
  __m128i v4; // xmm0
  unsigned int v5; // esi
  __int64 v6; // rdi
  __int64 v7; // r14
  __int32 v8; // edx
  int v9; // r11d
  unsigned int i; // eax
  int *v11; // r8
  int v12; // r10d
  unsigned int v13; // eax
  __int64 v14; // rax
  int v16; // eax
  int v17; // ecx
  const __m128i *v18; // r13
  __m128i v19; // xmm1
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // rdx
  unsigned __int64 v23; // rcx
  unsigned __int64 v24; // rsi
  int v25; // eax
  __int64 v26; // rsi
  __m128i *v27; // rsi
  __int64 m128i_i64; // rdi
  __m128i *v29; // rsi
  bool v30; // zf
  unsigned __int64 v31; // r12
  __int64 v32; // rdi
  __m128i v33; // [rsp+0h] [rbp-190h] BYREF
  int v34; // [rsp+10h] [rbp-180h]
  __int64 v35; // [rsp+20h] [rbp-170h] BYREF
  _BYTE *v36; // [rsp+28h] [rbp-168h]
  __int64 v37; // [rsp+30h] [rbp-160h]
  int v38; // [rsp+38h] [rbp-158h]
  char v39; // [rsp+3Ch] [rbp-154h]
  _BYTE v40[128]; // [rsp+40h] [rbp-150h] BYREF
  __m128i v41; // [rsp+C0h] [rbp-D0h] BYREF
  char v42[8]; // [rsp+D0h] [rbp-C0h] BYREF
  unsigned __int64 v43; // [rsp+D8h] [rbp-B8h]
  char v44; // [rsp+ECh] [rbp-A4h]
  char v45[160]; // [rsp+F0h] [rbp-A0h] BYREF

  v4 = _mm_loadu_si128(a2);
  v5 = *(_DWORD *)(a1 + 24);
  v34 = 0;
  v41 = v4;
  v33 = v4;
  if ( !v5 )
  {
    ++*(_QWORD *)a1;
    v41.m128i_i64[0] = 0;
    goto LABEL_35;
  }
  v6 = *(_QWORD *)(a1 + 8);
  v7 = 0;
  v8 = v33.m128i_i32[0];
  v9 = 1;
  for ( i = (v5 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned __int64)(unsigned int)(37 * v33.m128i_i32[0]) << 32)
              | ((unsigned __int32)v33.m128i_i32[2] >> 9) ^ ((unsigned __int32)v33.m128i_i32[2] >> 4))) >> 31)
           ^ (484763065 * (((unsigned __int32)v33.m128i_i32[2] >> 9) ^ ((unsigned __int32)v33.m128i_i32[2] >> 4))));
        ;
        i = (v5 - 1) & v13 )
  {
    v11 = (int *)(v6 + 24LL * i);
    v12 = *v11;
    if ( *v11 == v33.m128i_i32[0] && *((_QWORD *)v11 + 1) == v33.m128i_i64[1] )
    {
      v14 = (unsigned int)v11[4];
      return *(_QWORD *)(a1 + 32) + 176 * v14 + 16;
    }
    if ( v12 == 0x7FFFFFFF )
      break;
    if ( v12 == 0x80000000 && *((_QWORD *)v11 + 1) == -8192 && !v7 )
      v7 = v6 + 24LL * i;
LABEL_9:
    v13 = v9 + i;
    ++v9;
  }
  if ( *((_QWORD *)v11 + 1) != -4096 )
    goto LABEL_9;
  v16 = *(_DWORD *)(a1 + 16);
  if ( !v7 )
    v7 = (__int64)v11;
  ++*(_QWORD *)a1;
  v17 = v16 + 1;
  v41.m128i_i64[0] = v7;
  if ( 4 * (v16 + 1) >= 3 * v5 )
  {
LABEL_35:
    v18 = &v41;
    sub_34F9370(a1, 2 * v5);
    goto LABEL_36;
  }
  v18 = &v41;
  if ( v5 - *(_DWORD *)(a1 + 20) - v17 > v5 >> 3 )
    goto LABEL_18;
  sub_34F9370(a1, v5);
LABEL_36:
  sub_34F64E0(a1, v33.m128i_i32, &v41);
  v8 = v33.m128i_i32[0];
  v7 = v41.m128i_i64[0];
  v17 = *(_DWORD *)(a1 + 16) + 1;
LABEL_18:
  *(_DWORD *)(a1 + 16) = v17;
  if ( *(_DWORD *)v7 != 0x7FFFFFFF || *(_QWORD *)(v7 + 8) != -4096 )
    --*(_DWORD *)(a1 + 20);
  *(_DWORD *)v7 = v8;
  v35 = 0;
  *(_QWORD *)(v7 + 8) = v33.m128i_i64[1];
  v36 = v40;
  *(_DWORD *)(v7 + 16) = v34;
  v19 = _mm_loadu_si128(a2);
  v37 = 16;
  v38 = 0;
  v39 = 1;
  v41 = v19;
  sub_C8CF70((__int64)v42, v45, 16, (__int64)v40, (__int64)&v35);
  v22 = *(unsigned int *)(a1 + 40);
  v23 = *(unsigned int *)(a1 + 44);
  v24 = v22 + 1;
  v25 = *(_DWORD *)(a1 + 40);
  if ( v22 + 1 > v23 )
  {
    v31 = *(_QWORD *)(a1 + 32);
    v32 = a1 + 32;
    if ( v31 > (unsigned __int64)&v41 || (unsigned __int64)&v41 >= v31 + 176 * v22 )
    {
      sub_34F6CB0(v32, v24, v22, v23, v20, v21);
      v22 = *(unsigned int *)(a1 + 40);
      v26 = *(_QWORD *)(a1 + 32);
      v25 = *(_DWORD *)(a1 + 40);
    }
    else
    {
      sub_34F6CB0(v32, v24, v22, v23, v20, v21);
      v26 = *(_QWORD *)(a1 + 32);
      v22 = *(unsigned int *)(a1 + 40);
      v18 = (__m128i *)((char *)&v41 + v26 - v31);
      v25 = *(_DWORD *)(a1 + 40);
    }
  }
  else
  {
    v26 = *(_QWORD *)(a1 + 32);
  }
  v27 = (__m128i *)(176 * v22 + v26);
  if ( v27 )
  {
    m128i_i64 = (__int64)v27[1].m128i_i64;
    v29 = v27 + 3;
    v29[-3] = _mm_loadu_si128(v18);
    sub_C8CF70(m128i_i64, v29, 16, (__int64)v18[3].m128i_i64, (__int64)v18[1].m128i_i64);
    v25 = *(_DWORD *)(a1 + 40);
  }
  v30 = v44 == 0;
  *(_DWORD *)(a1 + 40) = v25 + 1;
  if ( v30 )
    _libc_free(v43);
  if ( !v39 )
    _libc_free((unsigned __int64)v36);
  v14 = (unsigned int)(*(_DWORD *)(a1 + 40) - 1);
  *(_DWORD *)(v7 + 16) = v14;
  return *(_QWORD *)(a1 + 32) + 176 * v14 + 16;
}
