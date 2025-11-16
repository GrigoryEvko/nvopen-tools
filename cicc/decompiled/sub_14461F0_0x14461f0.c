// Function: sub_14461F0
// Address: 0x14461f0
//
void __fastcall sub_14461F0(__int64 a1, _QWORD *a2)
{
  char *v4; // rdi
  __int64 v5; // rdx
  _BYTE *v6; // rsi
  _BYTE *v7; // r8
  unsigned __int64 v8; // r14
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 v11; // rdx
  _BYTE *v12; // rax
  char v13; // cl
  __int64 v14; // rcx
  __int64 v15; // r8
  unsigned __int64 v16; // r14
  __int64 v17; // rax
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // rdx
  __int64 v20; // rax
  char v21; // si
  unsigned __int64 v22; // rax
  __int64 v23; // rdx
  _QWORD *v24; // r8
  unsigned __int64 v25; // rax
  char v26; // si
  char v27; // r8
  bool v28; // si
  char v29[8]; // [rsp+0h] [rbp-220h] BYREF
  __int64 v30; // [rsp+8h] [rbp-218h]
  unsigned __int64 v31; // [rsp+10h] [rbp-210h]
  char v32[64]; // [rsp+28h] [rbp-1F8h] BYREF
  __int64 v33; // [rsp+68h] [rbp-1B8h]
  __int64 v34; // [rsp+70h] [rbp-1B0h]
  unsigned __int64 v35; // [rsp+78h] [rbp-1A8h]
  char v36[8]; // [rsp+80h] [rbp-1A0h] BYREF
  __int64 v37; // [rsp+88h] [rbp-198h]
  unsigned __int64 v38; // [rsp+90h] [rbp-190h]
  _BYTE v39[64]; // [rsp+A8h] [rbp-178h] BYREF
  unsigned __int64 v40; // [rsp+E8h] [rbp-138h]
  unsigned __int64 i; // [rsp+F0h] [rbp-130h]
  unsigned __int64 v42; // [rsp+F8h] [rbp-128h]
  _QWORD v43[2]; // [rsp+100h] [rbp-120h] BYREF
  unsigned __int64 v44; // [rsp+110h] [rbp-110h]
  _BYTE *v45; // [rsp+168h] [rbp-B8h]
  _BYTE *v46; // [rsp+170h] [rbp-B0h]
  __int64 v47; // [rsp+178h] [rbp-A8h]
  char v48[8]; // [rsp+180h] [rbp-A0h] BYREF
  __int64 v49; // [rsp+188h] [rbp-98h]
  unsigned __int64 v50; // [rsp+190h] [rbp-90h]
  __int64 v51; // [rsp+1E8h] [rbp-38h]
  __int64 v52; // [rsp+1F0h] [rbp-30h]
  __int64 v53; // [rsp+1F8h] [rbp-28h]

  sub_1445F60(v43, a2);
  v4 = v29;
  sub_16CCCB0(v29, v32, v43);
  v6 = v46;
  v7 = v45;
  v33 = 0;
  v34 = 0;
  v35 = 0;
  v8 = v46 - v45;
  if ( v46 == v45 )
  {
    v10 = 0;
  }
  else
  {
    if ( v8 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_61;
    v9 = sub_22077B0(v46 - v45);
    v6 = v46;
    v7 = v45;
    v10 = v9;
  }
  v33 = v10;
  v34 = v10;
  v35 = v10 + v8;
  if ( v7 != v6 )
  {
    v11 = v10;
    v12 = v7;
    do
    {
      if ( v11 )
      {
        *(_QWORD *)v11 = *(_QWORD *)v12;
        v13 = v12[32];
        *(_BYTE *)(v11 + 32) = v13;
        if ( v13 )
        {
          *(__m128i *)(v11 + 8) = _mm_loadu_si128((const __m128i *)(v12 + 8));
          *(_QWORD *)(v11 + 24) = *((_QWORD *)v12 + 3);
        }
      }
      v12 += 40;
      v11 += 40;
    }
    while ( v12 != v6 );
    v10 += 8 * ((unsigned __int64)(v12 - 40 - v7) >> 3) + 40;
  }
  v34 = v10;
  v4 = v36;
  v6 = v39;
  sub_16CCCB0(v36, v39, v48);
  v14 = v52;
  v15 = v51;
  v40 = 0;
  i = 0;
  v42 = 0;
  v16 = v52 - v51;
  if ( v52 == v51 )
  {
    v18 = 0;
    goto LABEL_14;
  }
  if ( v16 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_61:
    sub_4261EA(v4, v6, v5);
  v17 = sub_22077B0(v52 - v51);
  v14 = v52;
  v15 = v51;
  v18 = v17;
LABEL_14:
  v40 = v18;
  i = v18;
  v42 = v18 + v16;
  if ( v14 == v15 )
  {
    v22 = v18;
  }
  else
  {
    v19 = v18;
    v20 = v15;
    do
    {
      if ( v19 )
      {
        *(_QWORD *)v19 = *(_QWORD *)v20;
        v21 = *(_BYTE *)(v20 + 32);
        *(_BYTE *)(v19 + 32) = v21;
        if ( v21 )
        {
          *(__m128i *)(v19 + 8) = _mm_loadu_si128((const __m128i *)(v20 + 8));
          *(_QWORD *)(v19 + 24) = *(_QWORD *)(v20 + 24);
        }
      }
      v20 += 40;
      v19 += 40LL;
    }
    while ( v14 != v20 );
    v22 = v18 + 8 * ((unsigned __int64)(v14 - 40 - v15) >> 3) + 40;
  }
  for ( i = v22; ; v22 = i )
  {
    v23 = v33;
    if ( v34 - v33 != v22 - v18 )
      goto LABEL_25;
    if ( v33 == v34 )
      break;
    v25 = v18;
    while ( *(_QWORD *)v23 == *(_QWORD *)v25 )
    {
      v26 = *(_BYTE *)(v23 + 32);
      v27 = *(_BYTE *)(v25 + 32);
      if ( v26 && v27 )
      {
        if ( ((*(__int64 *)(v23 + 8) >> 1) & 3) != 0 )
          v28 = ((*(__int64 *)(v25 + 8) >> 1) & 3) == ((*(__int64 *)(v23 + 8) >> 1) & 3);
        else
          v28 = *(_DWORD *)(v23 + 24) == *(_DWORD *)(v25 + 24);
        if ( !v28 )
          break;
        v23 += 40;
        v25 += 40LL;
        if ( v34 == v23 )
          goto LABEL_37;
      }
      else
      {
        if ( v27 != v26 )
          break;
        v23 += 40;
        v25 += 40LL;
        if ( v34 == v23 )
          goto LABEL_37;
      }
    }
LABEL_25:
    v24 = *(_QWORD **)(v34 - 40);
    if ( (*v24 & 4) != 0 )
    {
      sub_14461F0(a1, *(_QWORD *)(v34 - 40));
    }
    else if ( a2 != (_QWORD *)sub_1443F20(a1, *v24 & 0xFFFFFFFFFFFFFFF8LL) )
    {
      sub_16BD130("BB map does not match region nesting", 1);
    }
    sub_1445BC0((__int64)v29);
    v18 = v40;
  }
LABEL_37:
  if ( v18 )
    j_j___libc_free_0(v18, v42 - v18);
  if ( v38 != v37 )
    _libc_free(v38);
  if ( v33 )
    j_j___libc_free_0(v33, v35 - v33);
  if ( v31 != v30 )
    _libc_free(v31);
  if ( v51 )
    j_j___libc_free_0(v51, v53 - v51);
  if ( v50 != v49 )
    _libc_free(v50);
  if ( v45 )
    j_j___libc_free_0(v45, v47 - (_QWORD)v45);
  if ( v44 != v43[1] )
    _libc_free(v44);
}
