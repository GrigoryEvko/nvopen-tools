// Function: sub_391F970
// Address: 0x391f970
//
void __fastcall sub_391F970(__int64 a1, unsigned int a2, _BYTE *a3, __int64 a4, __int64 a5, unsigned __int64 a6)
{
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rdi
  unsigned __int64 *v10; // rax
  char *v11; // rcx
  unsigned __int64 *v12; // rdx
  __int64 v13; // rbx
  unsigned __int64 v14; // r14
  char v15; // si
  char v16; // al
  char *v17; // rax
  __int64 v18; // rbx
  unsigned __int64 v19; // r14
  char v20; // si
  char v21; // al
  char *v22; // rax
  __int64 v23; // rbx
  unsigned __int64 v24; // r13
  unsigned int v25; // eax
  __int64 v26; // rdi
  int v27; // esi
  unsigned __int64 v28; // r14
  _BYTE *v29; // rax
  __int64 v30; // r15
  char v31; // si
  char v32; // al
  char *v33; // rax
  __int64 v34; // r15
  char v35; // si
  char v36; // al
  char *v37; // rax
  unsigned int v38; // eax
  __int64 v39; // r15
  __int64 v40; // r14
  char v41; // r13
  char v42; // si
  char *v43; // rax
  char v44; // al
  __int64 v46; // [rsp+18h] [rbp-B8h]
  _QWORD v47[4]; // [rsp+20h] [rbp-B0h] BYREF
  _BYTE *v48; // [rsp+40h] [rbp-90h] BYREF
  size_t v49; // [rsp+48h] [rbp-88h]
  _QWORD v50[2]; // [rsp+50h] [rbp-80h] BYREF
  char *v51; // [rsp+60h] [rbp-70h] BYREF
  size_t v52; // [rsp+68h] [rbp-68h]
  _QWORD v53[2]; // [rsp+70h] [rbp-60h] BYREF
  char *v54; // [rsp+80h] [rbp-50h]
  size_t v55; // [rsp+88h] [rbp-48h]
  _OWORD v56[4]; // [rsp+90h] [rbp-40h] BYREF

  if ( !a6 )
    return;
  if ( a3 )
  {
    v51 = (char *)v53;
    sub_3919930((__int64 *)&v51, a3, (__int64)&a3[a4]);
  }
  else
  {
    v52 = 0;
    v51 = (char *)v53;
    LOBYTE(v53[0]) = 0;
  }
  v48 = v50;
  sub_3919930((__int64 *)&v48, "reloc.", (__int64)"");
  v8 = 15;
  v9 = 15;
  if ( v48 != (_BYTE *)v50 )
    v9 = v50[0];
  if ( v49 + v52 <= v9 )
    goto LABEL_10;
  if ( v51 != (char *)v53 )
    v8 = v53[0];
  if ( v49 + v52 <= v8 )
  {
    v10 = sub_2241130((unsigned __int64 *)&v51, 0, 0, v48, v49);
    v54 = (char *)v56;
    v11 = (char *)*v10;
    v12 = v10 + 2;
    if ( (unsigned __int64 *)*v10 != v10 + 2 )
      goto LABEL_11;
  }
  else
  {
LABEL_10:
    v10 = sub_2241490((unsigned __int64 *)&v48, v51, v52);
    v54 = (char *)v56;
    v11 = (char *)*v10;
    v12 = v10 + 2;
    if ( (unsigned __int64 *)*v10 != v10 + 2 )
    {
LABEL_11:
      v54 = v11;
      *(_QWORD *)&v56[0] = v10[2];
      goto LABEL_12;
    }
  }
  v56[0] = _mm_loadu_si128((const __m128i *)v10 + 1);
LABEL_12:
  v55 = v10[1];
  *v10 = (unsigned __int64)v12;
  v10[1] = 0;
  *((_BYTE *)v10 + 16) = 0;
  sub_391B490(a1, (__int64)v47, v54, v55);
  if ( v54 != (char *)v56 )
    j_j___libc_free_0((unsigned __int64)v54);
  if ( v48 != (_BYTE *)v50 )
    j_j___libc_free_0((unsigned __int64)v48);
  if ( v51 != (char *)v53 )
    j_j___libc_free_0((unsigned __int64)v51);
  v13 = *(_QWORD *)(a1 + 8);
  v14 = a2;
  do
  {
    while ( 1 )
    {
      v15 = v14 & 0x7F;
      v16 = v14 & 0x7F | 0x80;
      v14 >>= 7;
      if ( v14 )
        v15 = v16;
      v17 = *(char **)(v13 + 24);
      if ( (unsigned __int64)v17 >= *(_QWORD *)(v13 + 16) )
        break;
      *(_QWORD *)(v13 + 24) = v17 + 1;
      *v17 = v15;
      if ( !v14 )
        goto LABEL_24;
    }
    sub_16E7DE0(v13, v15);
  }
  while ( v14 );
LABEL_24:
  v18 = *(_QWORD *)(a1 + 8);
  v19 = a6;
  do
  {
    while ( 1 )
    {
      v20 = v19 & 0x7F;
      v21 = v19 & 0x7F | 0x80;
      v19 >>= 7;
      if ( v19 )
        v20 = v21;
      v22 = *(char **)(v18 + 24);
      if ( (unsigned __int64)v22 >= *(_QWORD *)(v18 + 16) )
        break;
      *(_QWORD *)(v18 + 24) = v22 + 1;
      *v22 = v20;
      if ( !v19 )
        goto LABEL_30;
    }
    sub_16E7DE0(v18, v20);
  }
  while ( v19 );
LABEL_30:
  v23 = a5;
  v46 = a5 + 40 * a6;
  if ( v46 != a5 )
  {
    while ( 1 )
    {
      v24 = *(_QWORD *)(*(_QWORD *)(v23 + 32) + 184LL) + *(_QWORD *)v23;
      v25 = sub_391F690(a1, v23);
      v26 = *(_QWORD *)(a1 + 8);
      v27 = *(_DWORD *)(v23 + 24);
      v28 = v25;
      v29 = *(_BYTE **)(v26 + 24);
      if ( (unsigned __int64)v29 >= *(_QWORD *)(v26 + 16) )
      {
        sub_16E7DE0(v26, v27);
      }
      else
      {
        *(_QWORD *)(v26 + 24) = v29 + 1;
        *v29 = v27;
      }
      v30 = *(_QWORD *)(a1 + 8);
      do
      {
        while ( 1 )
        {
          v31 = v24 & 0x7F;
          v32 = v24 & 0x7F | 0x80;
          v24 >>= 7;
          if ( v24 )
            v31 = v32;
          v33 = *(char **)(v30 + 24);
          if ( (unsigned __int64)v33 >= *(_QWORD *)(v30 + 16) )
            break;
          *(_QWORD *)(v30 + 24) = v33 + 1;
          *v33 = v31;
          if ( !v24 )
            goto LABEL_39;
        }
        sub_16E7DE0(v30, v31);
      }
      while ( v24 );
LABEL_39:
      v34 = *(_QWORD *)(a1 + 8);
      do
      {
        while ( 1 )
        {
          v35 = v28 & 0x7F;
          v36 = v28 & 0x7F | 0x80;
          v28 >>= 7;
          if ( v28 )
            v35 = v36;
          v37 = *(char **)(v34 + 24);
          if ( (unsigned __int64)v37 >= *(_QWORD *)(v34 + 16) )
            break;
          *(_QWORD *)(v34 + 24) = v37 + 1;
          *v37 = v35;
          if ( !v28 )
            goto LABEL_45;
        }
        sub_16E7DE0(v34, v35);
      }
      while ( v28 );
LABEL_45:
      v38 = *(_DWORD *)(v23 + 24);
      if ( v38 <= 5 )
      {
        if ( v38 > 2 )
          break;
        goto LABEL_47;
      }
      if ( v38 - 8 <= 1 )
        break;
LABEL_47:
      v23 += 40;
      if ( v46 == v23 )
        goto LABEL_48;
    }
    v39 = *(_QWORD *)(a1 + 8);
    v40 = *(_QWORD *)(v23 + 16);
    while ( 1 )
    {
      v44 = v40;
      v42 = v40 & 0x7F;
      v40 >>= 7;
      if ( !v40 )
        break;
      if ( v40 != -1 )
        goto LABEL_52;
      v41 = 0;
      if ( (v44 & 0x40) == 0 )
        goto LABEL_52;
      v43 = *(char **)(v39 + 24);
      if ( (unsigned __int64)v43 >= *(_QWORD *)(v39 + 16) )
      {
LABEL_60:
        sub_16E7DE0(v39, v42);
        goto LABEL_55;
      }
LABEL_54:
      *(_QWORD *)(v39 + 24) = v43 + 1;
      *v43 = v42;
LABEL_55:
      if ( !v41 )
        goto LABEL_47;
    }
    v41 = 0;
    if ( (v44 & 0x40) != 0 )
    {
LABEL_52:
      v42 |= 0x80u;
      v41 = 1;
    }
    v43 = *(char **)(v39 + 24);
    if ( (unsigned __int64)v43 >= *(_QWORD *)(v39 + 16) )
      goto LABEL_60;
    goto LABEL_54;
  }
LABEL_48:
  sub_3919EA0(a1, v47);
}
