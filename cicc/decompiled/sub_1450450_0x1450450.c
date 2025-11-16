// Function: sub_1450450
// Address: 0x1450450
//
void __fastcall sub_1450450(__int64 *a1)
{
  char *v2; // rdi
  __int64 v3; // rdx
  __int64 v4; // rsi
  __int64 v5; // r8
  unsigned __int64 v6; // r13
  __int64 v7; // rax
  __int64 v8; // rdi
  __int64 v9; // rdx
  __int64 v10; // rax
  char v11; // cl
  __int64 v12; // r8
  unsigned __int64 v13; // r13
  __int64 v14; // rax
  unsigned __int64 v15; // rdi
  unsigned __int64 v16; // rdx
  __int64 v17; // rax
  char v18; // cl
  unsigned __int64 v19; // rax
  __int64 v20; // rdx
  unsigned __int64 v21; // rax
  char v22; // si
  char v23[8]; // [rsp+0h] [rbp-220h] BYREF
  __int64 v24; // [rsp+8h] [rbp-218h]
  unsigned __int64 v25; // [rsp+10h] [rbp-210h]
  char v26[64]; // [rsp+28h] [rbp-1F8h] BYREF
  __int64 v27; // [rsp+68h] [rbp-1B8h]
  __int64 v28; // [rsp+70h] [rbp-1B0h]
  unsigned __int64 v29; // [rsp+78h] [rbp-1A8h]
  char v30[8]; // [rsp+80h] [rbp-1A0h] BYREF
  __int64 v31; // [rsp+88h] [rbp-198h]
  unsigned __int64 v32; // [rsp+90h] [rbp-190h]
  char v33[64]; // [rsp+A8h] [rbp-178h] BYREF
  unsigned __int64 v34; // [rsp+E8h] [rbp-138h]
  unsigned __int64 i; // [rsp+F0h] [rbp-130h]
  unsigned __int64 v36; // [rsp+F8h] [rbp-128h]
  _QWORD v37[2]; // [rsp+100h] [rbp-120h] BYREF
  unsigned __int64 v38; // [rsp+110h] [rbp-110h]
  __int64 v39; // [rsp+168h] [rbp-B8h]
  __int64 v40; // [rsp+170h] [rbp-B0h]
  __int64 v41; // [rsp+178h] [rbp-A8h]
  char v42[8]; // [rsp+180h] [rbp-A0h] BYREF
  __int64 v43; // [rsp+188h] [rbp-98h]
  unsigned __int64 v44; // [rsp+190h] [rbp-90h]
  __int64 v45; // [rsp+1E8h] [rbp-38h]
  __int64 v46; // [rsp+1F0h] [rbp-30h]
  __int64 v47; // [rsp+1F8h] [rbp-28h]

  sub_144ED80(v37, a1[1]);
  v2 = v23;
  sub_16CCCB0(v23, v26, v37);
  v4 = v40;
  v5 = v39;
  v27 = 0;
  v28 = 0;
  v29 = 0;
  v6 = v40 - v39;
  if ( v40 == v39 )
  {
    v8 = 0;
  }
  else
  {
    if ( v6 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_54;
    v7 = sub_22077B0(v40 - v39);
    v4 = v40;
    v5 = v39;
    v8 = v7;
  }
  v27 = v8;
  v28 = v8;
  v29 = v8 + v6;
  if ( v5 != v4 )
  {
    v9 = v8;
    v10 = v5;
    do
    {
      if ( v9 )
      {
        *(_QWORD *)v9 = *(_QWORD *)v10;
        v11 = *(_BYTE *)(v10 + 32);
        *(_BYTE *)(v9 + 32) = v11;
        if ( v11 )
        {
          *(__m128i *)(v9 + 8) = _mm_loadu_si128((const __m128i *)(v10 + 8));
          *(_QWORD *)(v9 + 24) = *(_QWORD *)(v10 + 24);
        }
      }
      v10 += 40;
      v9 += 40;
    }
    while ( v10 != v4 );
    v8 += 8 * ((unsigned __int64)(v10 - 40 - v5) >> 3) + 40;
  }
  v28 = v8;
  v2 = v30;
  sub_16CCCB0(v30, v33, v42);
  v4 = v46;
  v12 = v45;
  v34 = 0;
  i = 0;
  v36 = 0;
  v13 = v46 - v45;
  if ( v46 == v45 )
  {
    v15 = 0;
    goto LABEL_14;
  }
  if ( v13 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_54:
    sub_4261EA(v2, v4, v3);
  v14 = sub_22077B0(v46 - v45);
  v4 = v46;
  v12 = v45;
  v15 = v14;
LABEL_14:
  v34 = v15;
  i = v15;
  v36 = v15 + v13;
  if ( v4 == v12 )
  {
    v19 = v15;
  }
  else
  {
    v16 = v15;
    v17 = v12;
    do
    {
      if ( v16 )
      {
        *(_QWORD *)v16 = *(_QWORD *)v17;
        v18 = *(_BYTE *)(v17 + 32);
        *(_BYTE *)(v16 + 32) = v18;
        if ( v18 )
        {
          *(__m128i *)(v16 + 8) = _mm_loadu_si128((const __m128i *)(v17 + 8));
          *(_QWORD *)(v16 + 24) = *(_QWORD *)(v17 + 24);
        }
      }
      v17 += 40;
      v16 += 40LL;
    }
    while ( v17 != v4 );
    v19 = v15 + 8 * ((unsigned __int64)(v17 - 40 - v12) >> 3) + 40;
  }
  for ( i = v19; ; v19 = i )
  {
    v20 = v27;
    if ( v28 - v27 != v19 - v15 )
      goto LABEL_22;
    if ( v27 == v28 )
      break;
    v21 = v15;
    while ( *(_QWORD *)v20 == *(_QWORD *)v21 )
    {
      v22 = *(_BYTE *)(v21 + 32);
      if ( *(_BYTE *)(v20 + 32) )
      {
        if ( !v22 || *(_DWORD *)(v20 + 24) != *(_DWORD *)(v21 + 24) || *(_QWORD *)(v20 + 8) != *(_QWORD *)(v21 + 8) )
          break;
      }
      else if ( v22 )
      {
        break;
      }
      v20 += 40;
      v21 += 40LL;
      if ( v28 == v20 )
        goto LABEL_34;
    }
LABEL_22:
    sub_144F630(a1, *(char **)(v28 - 40));
    sub_144F0D0((__int64)v23);
    v15 = v34;
  }
LABEL_34:
  if ( v15 )
    j_j___libc_free_0(v15, v36 - v15);
  if ( v32 != v31 )
    _libc_free(v32);
  if ( v27 )
    j_j___libc_free_0(v27, v29 - v27);
  if ( v25 != v24 )
    _libc_free(v25);
  if ( v45 )
    j_j___libc_free_0(v45, v47 - v45);
  if ( v44 != v43 )
    _libc_free(v44);
  if ( v39 )
    j_j___libc_free_0(v39, v41 - v39);
  if ( v38 != v37[1] )
    _libc_free(v38);
}
