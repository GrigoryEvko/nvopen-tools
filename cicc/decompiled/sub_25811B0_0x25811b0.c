// Function: sub_25811B0
// Address: 0x25811b0
//
__int64 __fastcall sub_25811B0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rdi
  __m128i v7; // rax
  _BYTE *v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  bool v12; // bl
  bool v13; // r11
  __m128i *v14; // rcx
  char v15; // al
  unsigned int v16; // r12d
  const void **v17; // rbx
  unsigned __int64 v18; // r15
  __int64 v19; // rsi
  __int64 v20; // rbx
  __int64 v21; // r15
  const void **v22; // rbx
  unsigned __int64 v23; // r15
  __int64 v24; // rsi
  __int64 v25; // rbx
  __int64 v26; // r15
  __m128i v28; // rax
  char v29; // al
  __m128i v30; // rax
  const void **v31; // r12
  const void **v32; // rbx
  unsigned int v33; // edx
  const void **v34; // r12
  __int64 v35; // rax
  const void **v36; // rbx
  __int64 v37; // r13
  _DWORD *v38; // r15
  unsigned int v39; // edx
  __int64 v40; // rdi
  __int64 v41; // rdx
  __int64 v42; // rcx
  __int64 v43; // r8
  bool v44; // al
  const void **v45; // r12
  const void **v46; // rbx
  __int64 v47; // r13
  _DWORD *v48; // r15
  unsigned int v49; // edx
  bool v50; // [rsp+17h] [rbp-2C9h]
  unsigned __int64 v51; // [rsp+18h] [rbp-2C8h]
  unsigned __int64 v52; // [rsp+20h] [rbp-2C0h]
  const void ***v53; // [rsp+28h] [rbp-2B8h]
  char v54; // [rsp+4Dh] [rbp-293h] BYREF
  char v55; // [rsp+4Eh] [rbp-292h] BYREF
  char v56; // [rsp+4Fh] [rbp-291h] BYREF
  _BYTE *v57; // [rsp+50h] [rbp-290h]
  __int64 v58; // [rsp+58h] [rbp-288h]
  __int64 v59; // [rsp+60h] [rbp-280h] BYREF
  int v60; // [rsp+68h] [rbp-278h]
  __m128i v61; // [rsp+70h] [rbp-270h] BYREF
  __int64 v62; // [rsp+80h] [rbp-260h] BYREF
  __int64 v63; // [rsp+88h] [rbp-258h]
  __int64 v64; // [rsp+90h] [rbp-250h]
  __int64 v65; // [rsp+98h] [rbp-248h]
  const void **v66; // [rsp+A0h] [rbp-240h]
  __int64 v67; // [rsp+A8h] [rbp-238h]
  _BYTE v68[128]; // [rsp+B0h] [rbp-230h] BYREF
  __m128i v69; // [rsp+130h] [rbp-1B0h] BYREF
  __int64 v70; // [rsp+140h] [rbp-1A0h]
  __int64 v71; // [rsp+148h] [rbp-198h]
  const void **v72; // [rsp+150h] [rbp-190h]
  __int64 v73; // [rsp+158h] [rbp-188h]
  _BYTE v74[128]; // [rsp+160h] [rbp-180h] BYREF
  _BYTE v75[256]; // [rsp+1E0h] [rbp-100h] BYREF

  v53 = (const void ***)(a1 + 88);
  sub_2560F70((__int64)v75, a1 + 88);
  v5 = *(_QWORD *)(a3 - 64);
  v6 = *(_QWORD *)(a3 - 96);
  v54 = 0;
  v52 = v5;
  v51 = *(_QWORD *)(a3 - 32);
  v7.m128i_i64[0] = sub_250D2C0(v6, 0);
  v69 = v7;
  v8 = sub_2527570(a2, &v69, a1, &v54);
  v57 = v8;
  v58 = v9;
  v12 = 0;
  v13 = v9 & (v8 != 0);
  if ( v13 )
  {
    v40 = (__int64)v8;
    v13 = sub_AD7A80(v8, (__int64)&v69, v9, v10, v11);
    if ( !v13 )
    {
      v44 = sub_AD7890(v40, (__int64)&v69, v41, v42, v43);
      v13 = 0;
      v12 = v44;
    }
  }
  v55 = 0;
  v14 = (__m128i *)&v62;
  v66 = (const void **)v68;
  v56 = 0;
  v62 = 0;
  v63 = 0;
  v64 = 0;
  v65 = 0;
  v67 = 0x800000000LL;
  v69 = 0u;
  v70 = 0;
  v71 = 0;
  v72 = (const void **)v74;
  v73 = 0x800000000LL;
  if ( !v12 )
  {
    v50 = v13;
    v28.m128i_i64[0] = sub_250D2C0(v52, 0);
    v61 = v28;
    v29 = sub_2580850(a1, a2, &v61, &v62, &v55, 0);
    v14 = (__m128i *)&v62;
    v13 = v50;
    if ( !v29 )
      goto LABEL_39;
  }
  v15 = v55;
  if ( v13 )
  {
LABEL_4:
    if ( !v15 )
    {
      v45 = (const void **)v14[2].m128i_i64[0];
      v46 = &v45[2 * v14[2].m128i_u32[2]];
      if ( v46 != v45 )
      {
        v47 = a1;
        v48 = (_DWORD *)(a1 + 112);
        do
        {
          if ( *(_BYTE *)(v47 + 105) )
          {
            sub_2575FB0(v48, v45);
            v49 = *(_DWORD *)(v47 + 152);
            if ( v49 >= unk_4FEF868 )
              *(_BYTE *)(v47 + 105) = *(_BYTE *)(v47 + 104);
            else
              *(_BYTE *)(v47 + 288) &= v49 == 0;
          }
          v45 += 2;
        }
        while ( v46 != v45 );
      }
      goto LABEL_6;
    }
    goto LABEL_5;
  }
  v30.m128i_i64[0] = sub_250D2C0(v51, 0);
  v61 = v30;
  if ( (unsigned __int8)sub_2580850(a1, a2, &v61, &v69, &v56, 0) )
  {
    if ( v12 )
    {
      v15 = v56;
      v14 = &v69;
      goto LABEL_4;
    }
    if ( v55 && v56 )
    {
LABEL_5:
      *(_BYTE *)(a1 + 288) = *(_DWORD *)(a1 + 152) == 0;
LABEL_6:
      v16 = (unsigned __int8)sub_255BE50((__int64)v75, v53);
      goto LABEL_7;
    }
    v31 = v66;
    v32 = &v66[2 * (unsigned int)v67];
    if ( v32 != v66 )
    {
      while ( 1 )
      {
        if ( !*(_BYTE *)(a1 + 105) )
          goto LABEL_47;
        sub_2575FB0((_DWORD *)(a1 + 112), v31);
        v33 = *(_DWORD *)(a1 + 152);
        if ( v33 >= unk_4FEF868 )
        {
          *(_BYTE *)(a1 + 105) = *(_BYTE *)(a1 + 104);
LABEL_47:
          v31 += 2;
          if ( v32 == v31 )
            break;
        }
        else
        {
          v31 += 2;
          *(_BYTE *)(a1 + 288) &= v33 == 0;
          if ( v32 == v31 )
            break;
        }
      }
    }
    v34 = v72;
    v35 = 2LL * (unsigned int)v73;
    v36 = &v72[v35];
    if ( &v72[v35] == v72 )
      goto LABEL_6;
    v37 = a1;
    v38 = (_DWORD *)(a1 + 112);
    while ( 1 )
    {
      if ( !*(_BYTE *)(v37 + 105) )
        goto LABEL_54;
      sub_2575FB0(v38, v34);
      v39 = *(_DWORD *)(v37 + 152);
      if ( v39 >= unk_4FEF868 )
      {
        *(_BYTE *)(v37 + 105) = *(_BYTE *)(v37 + 104);
LABEL_54:
        v34 += 2;
        if ( v36 == v34 )
          goto LABEL_6;
      }
      else
      {
        v34 += 2;
        *(_BYTE *)(v37 + 288) &= v39 == 0;
        if ( v36 == v34 )
          goto LABEL_6;
      }
    }
  }
LABEL_39:
  v16 = 0;
  *(_BYTE *)(a1 + 105) = *(_BYTE *)(a1 + 104);
LABEL_7:
  v17 = v72;
  v18 = (unsigned __int64)&v72[2 * (unsigned int)v73];
  if ( v72 != (const void **)v18 )
  {
    do
    {
      v18 -= 16LL;
      if ( *(_DWORD *)(v18 + 8) > 0x40u && *(_QWORD *)v18 )
        j_j___libc_free_0_0(*(_QWORD *)v18);
    }
    while ( v17 != (const void **)v18 );
    v18 = (unsigned __int64)v72;
  }
  if ( (_BYTE *)v18 != v74 )
    _libc_free(v18);
  v19 = (unsigned int)v71;
  if ( (_DWORD)v71 )
  {
    v20 = v69.m128i_i64[1];
    v60 = 0;
    v59 = -1;
    v61.m128i_i32[2] = 0;
    v21 = v69.m128i_i64[1] + 16LL * (unsigned int)v71;
    v61.m128i_i64[0] = -2;
    do
    {
      if ( *(_DWORD *)(v20 + 8) > 0x40u && *(_QWORD *)v20 )
        j_j___libc_free_0_0(*(_QWORD *)v20);
      v20 += 16;
    }
    while ( v21 != v20 );
    sub_969240(v61.m128i_i64);
    sub_969240(&v59);
    v19 = (unsigned int)v71;
  }
  sub_C7D6A0(v69.m128i_i64[1], 16 * v19, 8);
  v22 = v66;
  v23 = (unsigned __int64)&v66[2 * (unsigned int)v67];
  if ( v66 != (const void **)v23 )
  {
    do
    {
      v23 -= 16LL;
      if ( *(_DWORD *)(v23 + 8) > 0x40u && *(_QWORD *)v23 )
        j_j___libc_free_0_0(*(_QWORD *)v23);
    }
    while ( v22 != (const void **)v23 );
    v23 = (unsigned __int64)v66;
  }
  if ( (_BYTE *)v23 != v68 )
    _libc_free(v23);
  v24 = (unsigned int)v65;
  if ( (_DWORD)v65 )
  {
    v25 = v63;
    v61.m128i_i32[2] = 0;
    v61.m128i_i64[0] = -1;
    v69.m128i_i32[2] = 0;
    v26 = v63 + 16LL * (unsigned int)v65;
    v69.m128i_i64[0] = -2;
    do
    {
      if ( *(_DWORD *)(v25 + 8) > 0x40u && *(_QWORD *)v25 )
        j_j___libc_free_0_0(*(_QWORD *)v25);
      v25 += 16;
    }
    while ( v26 != v25 );
    sub_969240(v69.m128i_i64);
    sub_969240(v61.m128i_i64);
    v24 = (unsigned int)v65;
  }
  sub_C7D6A0(v63, 16 * v24, 8);
  sub_25485A0((__int64)v75);
  return v16;
}
