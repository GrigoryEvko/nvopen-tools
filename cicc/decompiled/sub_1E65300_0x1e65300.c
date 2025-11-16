// Function: sub_1E65300
// Address: 0x1e65300
//
void __fastcall sub_1E65300(__int64 a1, _QWORD *a2)
{
  _BYTE *v4; // rsi
  _QWORD *v5; // rdi
  __int64 v6; // rcx
  __int64 v7; // rdx
  unsigned __int64 v8; // r14
  __int64 v9; // rax
  __int64 v10; // rcx
  char v11; // si
  __int64 v12; // rax
  __int64 v13; // rcx
  unsigned __int64 v14; // r14
  __int64 v15; // rax
  __int64 v16; // rdi
  __int64 v17; // rdx
  __int64 v18; // rax
  char v19; // si
  __int64 v20; // rdx
  _QWORD *v21; // r8
  __int64 v22; // rax
  char v23; // si
  _QWORD v24[2]; // [rsp+0h] [rbp-220h] BYREF
  unsigned __int64 v25; // [rsp+10h] [rbp-210h]
  _BYTE v26[64]; // [rsp+28h] [rbp-1F8h] BYREF
  __int64 v27; // [rsp+68h] [rbp-1B8h]
  __int64 v28; // [rsp+70h] [rbp-1B0h]
  unsigned __int64 v29; // [rsp+78h] [rbp-1A8h]
  _QWORD v30[2]; // [rsp+80h] [rbp-1A0h] BYREF
  unsigned __int64 v31; // [rsp+90h] [rbp-190h]
  _BYTE v32[64]; // [rsp+A8h] [rbp-178h] BYREF
  __int64 v33; // [rsp+E8h] [rbp-138h]
  __int64 i; // [rsp+F0h] [rbp-130h]
  unsigned __int64 v35; // [rsp+F8h] [rbp-128h]
  _QWORD v36[2]; // [rsp+100h] [rbp-120h] BYREF
  unsigned __int64 v37; // [rsp+110h] [rbp-110h]
  __int64 v38; // [rsp+168h] [rbp-B8h]
  __int64 v39; // [rsp+170h] [rbp-B0h]
  __int64 v40; // [rsp+178h] [rbp-A8h]
  char v41[8]; // [rsp+180h] [rbp-A0h] BYREF
  __int64 v42; // [rsp+188h] [rbp-98h]
  unsigned __int64 v43; // [rsp+190h] [rbp-90h]
  __int64 v44; // [rsp+1E8h] [rbp-38h]
  __int64 v45; // [rsp+1F0h] [rbp-30h]
  __int64 v46; // [rsp+1F8h] [rbp-28h]

  sub_1E65070(v36, a2);
  v4 = v26;
  v5 = v24;
  sub_16CCCB0(v24, (__int64)v26, (__int64)v36);
  v6 = v39;
  v7 = v38;
  v27 = 0;
  v28 = 0;
  v29 = 0;
  v8 = v39 - v38;
  if ( v39 == v38 )
  {
    v9 = 0;
  }
  else
  {
    if ( v8 > 0x7FFFFFFFFFFFFFE0LL )
      goto LABEL_60;
    v9 = sub_22077B0(v39 - v38);
    v6 = v39;
    v7 = v38;
  }
  v27 = v9;
  v28 = v9;
  v29 = v9 + v8;
  if ( v7 == v6 )
  {
    v10 = v9;
  }
  else
  {
    v10 = v9 + v6 - v7;
    do
    {
      if ( v9 )
      {
        *(_QWORD *)v9 = *(_QWORD *)v7;
        v11 = *(_BYTE *)(v7 + 24);
        *(_BYTE *)(v9 + 24) = v11;
        if ( v11 )
          *(__m128i *)(v9 + 8) = _mm_loadu_si128((const __m128i *)(v7 + 8));
      }
      v9 += 32;
      v7 += 32;
    }
    while ( v9 != v10 );
  }
  v5 = v30;
  v28 = v10;
  v4 = v32;
  sub_16CCCB0(v30, (__int64)v32, (__int64)v41);
  v12 = v45;
  v13 = v44;
  v33 = 0;
  i = 0;
  v35 = 0;
  v14 = v45 - v44;
  if ( v45 == v44 )
  {
    v16 = 0;
    goto LABEL_13;
  }
  if ( v14 > 0x7FFFFFFFFFFFFFE0LL )
LABEL_60:
    sub_4261EA(v5, v4, v7);
  v15 = sub_22077B0(v45 - v44);
  v13 = v44;
  v16 = v15;
  v12 = v45;
LABEL_13:
  v33 = v16;
  i = v16;
  v35 = v16 + v14;
  if ( v12 == v13 )
  {
    v18 = v16;
  }
  else
  {
    v17 = v16;
    v18 = v16 + v12 - v13;
    do
    {
      if ( v17 )
      {
        *(_QWORD *)v17 = *(_QWORD *)v13;
        v19 = *(_BYTE *)(v13 + 24);
        *(_BYTE *)(v17 + 24) = v19;
        if ( v19 )
          *(__m128i *)(v17 + 8) = _mm_loadu_si128((const __m128i *)(v13 + 8));
      }
      v17 += 32;
      v13 += 32;
    }
    while ( v18 != v17 );
  }
  for ( i = v18; ; v18 = i )
  {
    v20 = v27;
    if ( v28 - v27 != v18 - v16 )
      goto LABEL_23;
    if ( v27 == v28 )
      break;
    v22 = v16;
    while ( *(_QWORD *)v20 == *(_QWORD *)v22 )
    {
      v23 = *(_BYTE *)(v22 + 24);
      if ( *(_BYTE *)(v20 + 24) )
      {
        if ( !v23 )
          break;
        if ( ((*(__int64 *)(v20 + 8) >> 1) & 3) != 0 )
        {
          if ( ((*(__int64 *)(v20 + 8) >> 1) & 3) != ((*(__int64 *)(v22 + 8) >> 1) & 3) )
            break;
        }
        else if ( *(_QWORD *)(v20 + 16) != *(_QWORD *)(v22 + 16) )
        {
          break;
        }
        v20 += 32;
        v22 += 32;
        if ( v28 == v20 )
          goto LABEL_34;
      }
      else
      {
        if ( v23 )
          break;
        v20 += 32;
        v22 += 32;
        if ( v28 == v20 )
          goto LABEL_34;
      }
    }
LABEL_23:
    v21 = *(_QWORD **)(v28 - 32);
    if ( (*v21 & 4) != 0 )
    {
      sub_1E65300(a1, *(_QWORD *)(v28 - 32));
    }
    else if ( a2 != (_QWORD *)sub_1E633D0(a1, *v21 & 0xFFFFFFFFFFFFFFF8LL) )
    {
      sub_16BD130("BB map does not match region nesting", 1u);
    }
    sub_1E64D90((__int64)v24);
    v16 = v33;
  }
LABEL_34:
  if ( v16 )
    j_j___libc_free_0(v16, v35 - v16);
  if ( v31 != v30[1] )
    _libc_free(v31);
  if ( v27 )
    j_j___libc_free_0(v27, v29 - v27);
  if ( v25 != v24[1] )
    _libc_free(v25);
  if ( v44 )
    j_j___libc_free_0(v44, v46 - v44);
  if ( v43 != v42 )
    _libc_free(v43);
  if ( v38 )
    j_j___libc_free_0(v38, v40 - v38);
  if ( v37 != v36[1] )
    _libc_free(v37);
}
