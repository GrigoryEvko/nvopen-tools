// Function: sub_3203E00
// Address: 0x3203e00
//
void __fastcall sub_3203E00(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r15
  unsigned __int8 v7; // al
  __int64 v8; // rdi
  __int64 v9; // rdx
  __int64 v10; // r13
  __int64 v11; // r15
  _BYTE *i; // r12
  __int64 v13; // r12
  __int16 v14; // ax
  unsigned __int8 v15; // al
  __int64 *v16; // r14
  __int64 *v17; // r12
  __int64 j; // rcx
  __m128i *v19; // rsi
  char *v20; // rbx
  char *v21; // r13
  __int64 v22; // rax
  unsigned __int64 *v23; // rax
  unsigned __int64 v24; // r12
  __m128i *v25; // rsi
  __int64 v26; // r8
  __int64 v27; // r9
  __int64 v28; // rax
  __int64 v29; // [rsp+8h] [rbp-C8h]
  __m128i v30; // [rsp+10h] [rbp-C0h] BYREF
  __m128i v31; // [rsp+20h] [rbp-B0h] BYREF
  __int64 *v32; // [rsp+38h] [rbp-98h]
  __int64 *v33; // [rsp+40h] [rbp-90h]
  __int64 v34; // [rsp+58h] [rbp-78h]
  unsigned int v35; // [rsp+68h] [rbp-68h]
  char *v36; // [rsp+70h] [rbp-60h]
  int v37; // [rsp+78h] [rbp-58h]
  char v38; // [rsp+80h] [rbp-50h] BYREF
  unsigned __int64 v39; // [rsp+88h] [rbp-48h]

  v3 = a3 - 16;
  v7 = *(_BYTE *)(a3 - 16);
  if ( (v7 & 2) != 0 )
  {
    v8 = *(_QWORD *)(*(_QWORD *)(a3 - 32) + 16LL);
    if ( !v8 )
    {
      v10 = *(_QWORD *)(a3 + 32);
      goto LABEL_16;
    }
LABEL_3:
    sub_B91420(v8);
    if ( v9 )
    {
      v31 = (__m128i)(unsigned __int64)a3;
      v25 = *(__m128i **)(a2 + 32);
      if ( v25 == *(__m128i **)(a2 + 40) )
      {
        sub_31FC610(a2 + 24, v25, &v31);
      }
      else
      {
        if ( v25 )
        {
          *v25 = _mm_loadu_si128(&v31);
          v25 = *(__m128i **)(a2 + 32);
        }
        *(_QWORD *)(a2 + 32) = v25 + 1;
      }
      if ( (*(_BYTE *)(a3 + 21) & 0x10) != 0
        && sub_AF2DC0(a3)
        && (*(_BYTE *)sub_AF2DC0(a3) == 17 || *(_BYTE *)sub_AF2DC0(a3) == 18) )
      {
        v28 = *(unsigned int *)(a1 + 976);
        if ( v28 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 980) )
        {
          sub_C8D5F0(a1 + 968, (const void *)(a1 + 984), v28 + 1, 8u, v26, v27);
          v28 = *(unsigned int *)(a1 + 976);
        }
        *(_QWORD *)(*(_QWORD *)(a1 + 968) + 8 * v28) = a3;
        ++*(_DWORD *)(a1 + 976);
      }
      return;
    }
    v7 = *(_BYTE *)(a3 - 16);
    v10 = *(_QWORD *)(a3 + 32);
    if ( (v7 & 2) == 0 )
      goto LABEL_5;
LABEL_16:
    v11 = *(_QWORD *)(a3 - 32);
    goto LABEL_6;
  }
  v8 = *(_QWORD *)(a3 - 8LL * ((v7 >> 2) & 0xF));
  if ( v8 )
    goto LABEL_3;
  v10 = *(_QWORD *)(a3 + 32);
LABEL_5:
  v11 = v3 - 8LL * ((v7 >> 2) & 0xF);
LABEL_6:
  for ( i = *(_BYTE **)(v11 + 24); ; i = *(_BYTE **)(v13 + 24) )
  {
    v14 = sub_AF18C0((__int64)i);
    if ( v14 != 38 && v14 != 53 )
      break;
    v15 = *(i - 16);
    if ( (v15 & 2) != 0 )
      v13 = *((_QWORD *)i - 4);
    else
      v13 = (__int64)&i[-8 * ((v15 >> 2) & 0xF) - 16];
  }
  if ( *i == 14 )
  {
    sub_3203600((__int64)&v31, a1, (__int64)i);
    v16 = v33;
    v17 = v32;
    for ( j = a2 + 24; v16 != v17; *(_QWORD *)(a2 + 32) = v19 + 1 )
    {
      while ( 1 )
      {
        v19 = *(__m128i **)(a2 + 32);
        v30.m128i_i64[0] = *v17;
        v30.m128i_i64[1] = v10 + v17[1];
        if ( v19 != *(__m128i **)(a2 + 40) )
          break;
        v17 += 2;
        v29 = j;
        sub_31FC610(j, v19, &v30);
        j = v29;
        if ( v16 == v17 )
          goto LABEL_25;
      }
      if ( v19 )
      {
        *v19 = _mm_loadu_si128(&v30);
        v19 = *(__m128i **)(a2 + 32);
      }
      v17 += 2;
    }
LABEL_25:
    if ( v39 )
      j_j___libc_free_0(v39);
    v20 = v36;
    v21 = &v36[16 * v37];
    if ( v36 != v21 )
    {
      do
      {
        v22 = *((_QWORD *)v21 - 1);
        v21 -= 16;
        if ( v22 )
        {
          if ( (v22 & 4) != 0 )
          {
            v23 = (unsigned __int64 *)(v22 & 0xFFFFFFFFFFFFFFF8LL);
            v24 = (unsigned __int64)v23;
            if ( v23 )
            {
              if ( (unsigned __int64 *)*v23 != v23 + 2 )
                _libc_free(*v23);
              j_j___libc_free_0(v24);
            }
          }
        }
      }
      while ( v20 != v21 );
      v21 = v36;
    }
    if ( v21 != &v38 )
      _libc_free((unsigned __int64)v21);
    sub_C7D6A0(v34, 16LL * v35, 8);
    if ( v32 )
      j_j___libc_free_0((unsigned __int64)v32);
    if ( v31.m128i_i64[0] )
      j_j___libc_free_0(v31.m128i_u64[0]);
  }
}
