// Function: sub_2581BF0
// Address: 0x2581bf0
//
__int64 __fastcall sub_2581BF0(__int64 a1, __int64 a2, _BYTE *a3)
{
  __int64 v4; // rbx
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // r15
  __m128i v7; // rax
  __m128i v8; // rax
  unsigned int v9; // r12d
  _BYTE *v10; // rbx
  unsigned __int64 v11; // r13
  __int64 v12; // rsi
  __int64 v13; // rbx
  __int64 v14; // r13
  _BYTE *v15; // rbx
  unsigned __int64 v16; // r13
  __int64 v17; // rsi
  __int64 v18; // rbx
  __int64 v19; // r13
  __int64 v21; // r13
  _DWORD *v22; // rax
  __int64 v23; // rbx
  __int64 v24; // rdx
  unsigned __int64 v25; // r13
  char v26; // dl
  __int64 v27; // r13
  unsigned __int64 v28; // rbx
  _BYTE *v29; // r13
  __int64 v30; // r14
  unsigned int v31; // edx
  _BYTE *v32; // [rsp+0h] [rbp-300h]
  _DWORD *v33; // [rsp+8h] [rbp-2F8h]
  char v34; // [rsp+17h] [rbp-2E9h]
  __int64 v35; // [rsp+18h] [rbp-2E8h]
  _BYTE *v36; // [rsp+18h] [rbp-2E8h]
  __int64 v37; // [rsp+20h] [rbp-2E0h]
  _BYTE *v38; // [rsp+28h] [rbp-2D8h]
  char v39; // [rsp+28h] [rbp-2D8h]
  const void ***v40; // [rsp+30h] [rbp-2D0h]
  char v41; // [rsp+6Dh] [rbp-293h] BYREF
  char v42; // [rsp+6Eh] [rbp-292h] BYREF
  char v43; // [rsp+6Fh] [rbp-291h] BYREF
  __int64 v44; // [rsp+70h] [rbp-290h] BYREF
  unsigned int v45; // [rsp+78h] [rbp-288h]
  __m128i v46; // [rsp+80h] [rbp-280h] BYREF
  __m128i v47; // [rsp+90h] [rbp-270h] BYREF
  __int64 v48; // [rsp+A0h] [rbp-260h] BYREF
  __int64 v49; // [rsp+A8h] [rbp-258h]
  __int64 v50; // [rsp+B0h] [rbp-250h]
  __int64 v51; // [rsp+B8h] [rbp-248h]
  _BYTE *v52; // [rsp+C0h] [rbp-240h]
  __int64 v53; // [rsp+C8h] [rbp-238h]
  _BYTE v54[128]; // [rsp+D0h] [rbp-230h] BYREF
  __int64 v55; // [rsp+150h] [rbp-1B0h] BYREF
  __int64 v56; // [rsp+158h] [rbp-1A8h]
  __int64 v57; // [rsp+160h] [rbp-1A0h]
  __int64 v58; // [rsp+168h] [rbp-198h]
  _BYTE *v59; // [rsp+170h] [rbp-190h]
  __int64 v60; // [rsp+178h] [rbp-188h]
  _BYTE v61[128]; // [rsp+180h] [rbp-180h] BYREF
  _BYTE v62[256]; // [rsp+200h] [rbp-100h] BYREF

  v4 = a1;
  v40 = (const void ***)(a1 + 88);
  sub_2560F70((__int64)v62, a1 + 88);
  v5 = *((_QWORD *)a3 - 8);
  v6 = *((_QWORD *)a3 - 4);
  v52 = v54;
  v59 = v61;
  v41 = 0;
  v42 = 0;
  v48 = 0;
  v49 = 0;
  v50 = 0;
  v51 = 0;
  v53 = 0x800000000LL;
  v55 = 0;
  v56 = 0;
  v57 = 0;
  v58 = 0;
  v60 = 0x800000000LL;
  v7.m128i_i64[0] = sub_250D2C0(v5, 0);
  v46 = v7;
  if ( !(unsigned __int8)sub_2580850(v4, a2, &v46, &v48, &v41, 0)
    || (v8.m128i_i64[0] = sub_250D2C0(v6, 0), v47 = v8, !(unsigned __int8)sub_2580850(v4, a2, &v47, &v55, &v42, 0)) )
  {
    v9 = 0;
    *(_BYTE *)(v4 + 105) = *(_BYTE *)(v4 + 104);
    goto LABEL_4;
  }
  v45 = *(_DWORD *)(*(_QWORD *)(v5 + 8) + 8LL) >> 8;
  if ( v45 > 0x40 )
    sub_C43690((__int64)&v44, 0, 0);
  else
    v44 = 0;
  if ( v41 )
  {
    if ( v42 )
    {
      if ( !(unsigned __int8)sub_25763A0(v4, a3, (__int64)&v44, (unsigned __int64)&v44) )
        goto LABEL_61;
    }
    else
    {
      v27 = 16LL * (unsigned int)v60;
      v36 = &v59[v27];
      if ( &v59[v27] != v59 )
      {
        v37 = v4;
        v28 = (unsigned __int64)v59;
        while ( 1 )
        {
          v43 = 0;
          v46.m128i_i8[0] = 0;
          sub_254BC20((__int64)&v47, a3, (__int64)&v44, v28, &v43, &v46);
          if ( v46.m128i_i8[0] )
          {
            v4 = v37;
            sub_969240(v47.m128i_i64);
            goto LABEL_61;
          }
          if ( !v43 )
            sub_25761A0((__int64)v40, (const void **)&v47);
          v39 = *(_BYTE *)(v37 + 105);
          sub_969240(v47.m128i_i64);
          if ( !v39 )
            break;
          v28 += 16LL;
          if ( v36 == (_BYTE *)v28 )
            goto LABEL_64;
        }
        v4 = v37;
LABEL_61:
        v9 = 0;
        *(_BYTE *)(v4 + 105) = *(_BYTE *)(v4 + 104);
        goto LABEL_62;
      }
    }
  }
  else
  {
    v21 = 16LL * (unsigned int)v53;
    if ( v42 )
    {
      v29 = &v52[v21];
      if ( v29 != v52 )
      {
        v30 = (__int64)v52;
        while ( (unsigned __int8)sub_25763A0(v4, a3, v30, (unsigned __int64)&v44) )
        {
          v30 += 16;
          if ( v29 == (_BYTE *)v30 )
            goto LABEL_64;
        }
        v9 = 0;
        *(_BYTE *)(v4 + 105) = *(_BYTE *)(v4 + 104);
        goto LABEL_62;
      }
    }
    else
    {
      v32 = &v52[v21];
      if ( &v52[v21] != v52 )
      {
        v35 = v4;
        v22 = (_DWORD *)(v4 + 112);
        v23 = (__int64)v52;
        v33 = v22;
        while ( 1 )
        {
          v24 = 16LL * (unsigned int)v60;
          v25 = (unsigned __int64)v59;
          v38 = &v59[v24];
          if ( &v59[v24] != v59 )
            break;
LABEL_76:
          v23 += 16;
          if ( v32 == (_BYTE *)v23 )
            goto LABEL_64;
        }
        while ( 1 )
        {
          v43 = 0;
          v46.m128i_i8[0] = 0;
          sub_254BC20((__int64)&v47, a3, v23, v25, &v43, &v46);
          if ( v46.m128i_i8[0] )
          {
            if ( v47.m128i_i32[2] <= 0x40u )
              goto LABEL_75;
            v26 = 0;
          }
          else
          {
            v26 = *(_BYTE *)(v35 + 105);
            if ( v43 )
              goto LABEL_44;
            if ( v26 )
            {
              sub_2575FB0(v33, (const void **)&v47);
              v31 = *(_DWORD *)(v35 + 152);
              if ( v31 < unk_4FEF868 )
              {
                *(_BYTE *)(v35 + 288) &= v31 == 0;
                v26 = *(_BYTE *)(v35 + 105);
              }
              else
              {
                v26 = *(_BYTE *)(v35 + 104);
                *(_BYTE *)(v35 + 105) = v26;
              }
LABEL_44:
              if ( v47.m128i_i32[2] <= 0x40u )
                goto LABEL_45;
              goto LABEL_50;
            }
            if ( v47.m128i_i32[2] <= 0x40u )
            {
LABEL_75:
              v4 = v35;
              goto LABEL_61;
            }
          }
LABEL_50:
          if ( v47.m128i_i64[0] )
          {
            v34 = v26;
            j_j___libc_free_0_0(v47.m128i_u64[0]);
            v26 = v34;
          }
LABEL_45:
          if ( !v26 )
            goto LABEL_75;
          v25 += 16LL;
          if ( v38 == (_BYTE *)v25 )
            goto LABEL_76;
        }
      }
    }
  }
LABEL_64:
  v9 = (unsigned __int8)sub_255BE50((__int64)v62, v40);
LABEL_62:
  sub_969240(&v44);
LABEL_4:
  v10 = v59;
  v11 = (unsigned __int64)&v59[16 * (unsigned int)v60];
  if ( v59 != (_BYTE *)v11 )
  {
    do
    {
      v11 -= 16LL;
      if ( *(_DWORD *)(v11 + 8) > 0x40u && *(_QWORD *)v11 )
        j_j___libc_free_0_0(*(_QWORD *)v11);
    }
    while ( v10 != (_BYTE *)v11 );
    v11 = (unsigned __int64)v59;
  }
  if ( (_BYTE *)v11 != v61 )
    _libc_free(v11);
  v12 = (unsigned int)v58;
  if ( (_DWORD)v58 )
  {
    v13 = v56;
    v46.m128i_i32[2] = 0;
    v46.m128i_i64[0] = -1;
    v47.m128i_i32[2] = 0;
    v14 = v56 + 16LL * (unsigned int)v58;
    v47.m128i_i64[0] = -2;
    do
    {
      if ( *(_DWORD *)(v13 + 8) > 0x40u && *(_QWORD *)v13 )
        j_j___libc_free_0_0(*(_QWORD *)v13);
      v13 += 16;
    }
    while ( v14 != v13 );
    sub_969240(v47.m128i_i64);
    sub_969240(v46.m128i_i64);
    v12 = (unsigned int)v58;
  }
  sub_C7D6A0(v56, 16 * v12, 8);
  v15 = v52;
  v16 = (unsigned __int64)&v52[16 * (unsigned int)v53];
  if ( v52 != (_BYTE *)v16 )
  {
    do
    {
      v16 -= 16LL;
      if ( *(_DWORD *)(v16 + 8) > 0x40u && *(_QWORD *)v16 )
        j_j___libc_free_0_0(*(_QWORD *)v16);
    }
    while ( v15 != (_BYTE *)v16 );
    v16 = (unsigned __int64)v52;
  }
  if ( (_BYTE *)v16 != v54 )
    _libc_free(v16);
  v17 = (unsigned int)v51;
  if ( (_DWORD)v51 )
  {
    v18 = v49;
    v47.m128i_i32[2] = 0;
    v47.m128i_i64[0] = -1;
    LODWORD(v56) = 0;
    v19 = v49 + 16LL * (unsigned int)v51;
    v55 = -2;
    do
    {
      if ( *(_DWORD *)(v18 + 8) > 0x40u && *(_QWORD *)v18 )
        j_j___libc_free_0_0(*(_QWORD *)v18);
      v18 += 16;
    }
    while ( v19 != v18 );
    sub_969240(&v55);
    sub_969240(v47.m128i_i64);
    v17 = (unsigned int)v51;
  }
  sub_C7D6A0(v49, 16 * v17, 8);
  sub_25485A0((__int64)v62);
  return v9;
}
