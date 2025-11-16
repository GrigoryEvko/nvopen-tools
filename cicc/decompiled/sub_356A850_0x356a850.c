// Function: sub_356A850
// Address: 0x356a850
//
void __fastcall sub_356A850(__int64 a1, _QWORD *a2)
{
  _BYTE *v4; // rsi
  char *v5; // rdi
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // r8
  __int64 v10; // r9
  const __m128i *v11; // rcx
  const __m128i *v12; // rdx
  unsigned __int64 v13; // r14
  __m128i *v14; // rax
  __int64 v15; // rcx
  const __m128i *v16; // rax
  const __m128i *v17; // rcx
  unsigned __int64 v18; // r14
  __int64 v19; // rax
  unsigned __int64 v20; // rdi
  __m128i *v21; // rdx
  __m128i *v22; // rax
  unsigned __int64 v23; // rcx
  _QWORD *v24; // r8
  unsigned __int64 v25; // rax
  char v26; // si
  char v27[8]; // [rsp+0h] [rbp-210h] BYREF
  unsigned __int64 v28; // [rsp+8h] [rbp-208h]
  char v29; // [rsp+1Ch] [rbp-1F4h]
  _BYTE v30[64]; // [rsp+20h] [rbp-1F0h] BYREF
  __m128i *v31; // [rsp+60h] [rbp-1B0h]
  __int64 v32; // [rsp+68h] [rbp-1A8h]
  __int8 *v33; // [rsp+70h] [rbp-1A0h]
  char v34[8]; // [rsp+80h] [rbp-190h] BYREF
  unsigned __int64 v35; // [rsp+88h] [rbp-188h]
  char v36; // [rsp+9Ch] [rbp-174h]
  _BYTE v37[64]; // [rsp+A0h] [rbp-170h] BYREF
  unsigned __int64 v38; // [rsp+E0h] [rbp-130h]
  unsigned __int64 i; // [rsp+E8h] [rbp-128h]
  unsigned __int64 v40; // [rsp+F0h] [rbp-120h]
  _QWORD v41[3]; // [rsp+100h] [rbp-110h] BYREF
  char v42; // [rsp+11Ch] [rbp-F4h]
  const __m128i *v43; // [rsp+160h] [rbp-B0h]
  const __m128i *v44; // [rsp+168h] [rbp-A8h]
  char v45[8]; // [rsp+178h] [rbp-98h] BYREF
  unsigned __int64 v46; // [rsp+180h] [rbp-90h]
  char v47; // [rsp+194h] [rbp-7Ch]
  const __m128i *v48; // [rsp+1D8h] [rbp-38h]
  const __m128i *v49; // [rsp+1E0h] [rbp-30h]

  sub_356A570(v41, a2);
  v4 = v30;
  v5 = v27;
  sub_C8CD80((__int64)v27, (__int64)v30, (__int64)v41, v6, v7, v8);
  v11 = v44;
  v12 = v43;
  v31 = 0;
  v32 = 0;
  v33 = 0;
  v13 = (char *)v44 - (char *)v43;
  if ( v44 == v43 )
  {
    v14 = 0;
  }
  else
  {
    if ( v13 > 0x7FFFFFFFFFFFFFE0LL )
      goto LABEL_55;
    v14 = (__m128i *)sub_22077B0((char *)v44 - (char *)v43);
    v11 = v44;
    v12 = v43;
  }
  v31 = v14;
  v32 = (__int64)v14;
  v33 = &v14->m128i_i8[v13];
  if ( v12 == v11 )
  {
    v15 = (__int64)v14;
  }
  else
  {
    v15 = (__int64)v14->m128i_i64 + (char *)v11 - (char *)v12;
    do
    {
      if ( v14 )
      {
        *v14 = _mm_loadu_si128(v12);
        v14[1] = _mm_loadu_si128(v12 + 1);
      }
      v14 += 2;
      v12 += 2;
    }
    while ( v14 != (__m128i *)v15 );
  }
  v5 = v34;
  v32 = v15;
  v4 = v37;
  sub_C8CD80((__int64)v34, (__int64)v37, (__int64)v45, v15, v9, v10);
  v16 = v49;
  v17 = v48;
  v38 = 0;
  i = 0;
  v40 = 0;
  v18 = (char *)v49 - (char *)v48;
  if ( v49 == v48 )
  {
    v20 = 0;
    goto LABEL_12;
  }
  if ( v18 > 0x7FFFFFFFFFFFFFE0LL )
LABEL_55:
    sub_4261EA(v5, v4, v12);
  v19 = sub_22077B0((char *)v49 - (char *)v48);
  v17 = v48;
  v20 = v19;
  v16 = v49;
LABEL_12:
  v38 = v20;
  i = v20;
  v40 = v20 + v18;
  if ( v17 == v16 )
  {
    v22 = (__m128i *)v20;
  }
  else
  {
    v21 = (__m128i *)v20;
    v22 = (__m128i *)(v20 + (char *)v16 - (char *)v17);
    do
    {
      if ( v21 )
      {
        *v21 = _mm_loadu_si128(v17);
        v21[1] = _mm_loadu_si128(v17 + 1);
      }
      v21 += 2;
      v17 += 2;
    }
    while ( v22 != v21 );
  }
  for ( i = (unsigned __int64)v22; ; v22 = (__m128i *)i )
  {
    v23 = (unsigned __int64)v31;
    if ( (__m128i *)(v32 - (_QWORD)v31) != (__m128i *)((char *)v22 - v20) )
      goto LABEL_21;
    if ( v31 == (__m128i *)v32 )
      break;
    v25 = v20;
    while ( *(_QWORD *)v23 == *(_QWORD *)v25 )
    {
      v26 = *(_BYTE *)(v23 + 24);
      if ( v26 != *(_BYTE *)(v25 + 24) )
        break;
      if ( v26 )
      {
        if ( ((*(__int64 *)(v23 + 8) >> 1) & 3) != 0 )
        {
          if ( ((*(__int64 *)(v23 + 8) >> 1) & 3) != ((*(__int64 *)(v25 + 8) >> 1) & 3) )
            break;
        }
        else if ( *(_QWORD *)(v23 + 16) != *(_QWORD *)(v25 + 16) )
        {
          break;
        }
      }
      v23 += 32LL;
      v25 += 32LL;
      if ( v32 == v23 )
        goto LABEL_32;
    }
LABEL_21:
    v24 = *(_QWORD **)(v32 - 32);
    if ( (*v24 & 4) != 0 )
    {
      sub_356A850(a1, *(_QWORD *)(v32 - 32));
    }
    else if ( a2 != (_QWORD *)sub_3568610(a1, *v24 & 0xFFFFFFFFFFFFFFF8LL) )
    {
      sub_C64ED0("BB map does not match region nesting", 1u);
    }
    sub_3569CB0((__int64)v27);
    v20 = v38;
  }
LABEL_32:
  if ( v20 )
    j_j___libc_free_0(v20);
  if ( !v36 )
    _libc_free(v35);
  if ( v31 )
    j_j___libc_free_0((unsigned __int64)v31);
  if ( !v29 )
    _libc_free(v28);
  if ( v48 )
    j_j___libc_free_0((unsigned __int64)v48);
  if ( !v47 )
    _libc_free(v46);
  if ( v43 )
    j_j___libc_free_0((unsigned __int64)v43);
  if ( !v42 )
    _libc_free(v41[1]);
}
