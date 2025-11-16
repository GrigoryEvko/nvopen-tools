// Function: sub_15DF750
// Address: 0x15df750
//
__int64 __fastcall sub_15DF750(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  char v5; // al
  unsigned __int64 v6; // rcx
  __int8 *v7; // r8
  __m128i *v8; // rax
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // rcx
  __m128i *v12; // rax
  __m128i *v13; // rcx
  __m128i *v14; // rdx
  __int64 v15; // rcx
  unsigned __int64 v17; // rax
  unsigned __int64 v18; // rcx
  __int8 *v19; // r8
  __m128i *v20; // rax
  unsigned __int64 v21; // rax
  unsigned __int64 v22; // rdi
  unsigned __int64 v23; // rcx
  unsigned __int64 v24; // rax
  unsigned __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // rsi
  __int64 v28; // rcx
  __m128i *v29; // rax
  __int64 v30; // rcx
  __int64 v31; // rcx
  unsigned __int64 v32; // rbx
  __int64 v33; // rcx
  _QWORD *v34; // rbx
  _QWORD *i; // r14
  __int64 v36; // rcx
  unsigned __int64 v37; // rcx
  __int8 *v38; // r8
  __m128i *v39; // rax
  __int64 v40; // rcx
  unsigned __int64 v41; // rcx
  __int8 *v42; // r8
  __m128i *v43; // rax
  unsigned __int64 v44; // rax
  unsigned __int64 v45; // r9
  unsigned __int64 v46; // rax
  unsigned __int64 v47; // rax
  __int64 v48[2]; // [rsp+0h] [rbp-B0h] BYREF
  _QWORD v49[2]; // [rsp+10h] [rbp-A0h] BYREF
  __m128i *v50; // [rsp+20h] [rbp-90h] BYREF
  __int64 v51; // [rsp+28h] [rbp-88h]
  __m128i v52; // [rsp+30h] [rbp-80h] BYREF
  _QWORD *v53; // [rsp+40h] [rbp-70h] BYREF
  __int64 v54; // [rsp+48h] [rbp-68h]
  _QWORD v55[2]; // [rsp+50h] [rbp-60h] BYREF
  __m128i *v56; // [rsp+60h] [rbp-50h] BYREF
  __int64 v57; // [rsp+68h] [rbp-48h]
  __m128i v58; // [rsp+70h] [rbp-40h] BYREF

  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0;
  *(_BYTE *)(a1 + 16) = 0;
  v5 = *(_BYTE *)(a2 + 8);
  switch ( v5 )
  {
    case 15:
      sub_15DF750(&v53, *(_QWORD *)(a2 + 24));
      v6 = *(_DWORD *)(a2 + 8) >> 8;
      if ( *(_DWORD *)(a2 + 8) >> 8 )
      {
        v7 = &v58.m128i_i8[5];
        do
        {
          *--v7 = v6 % 0xA + 48;
          v17 = v6;
          v6 /= 0xAu;
        }
        while ( v17 > 9 );
      }
      else
      {
        v58.m128i_i8[4] = 48;
        v7 = &v58.m128i_i8[4];
      }
      v48[0] = (__int64)v49;
      sub_15DE6F0(v48, v7, (__int64)v58.m128i_i64 + 5);
      v8 = (__m128i *)sub_2241130(v48, 0, 0, "p", 1);
      v50 = &v52;
      if ( (__m128i *)v8->m128i_i64[0] == &v8[1] )
      {
        v52 = _mm_loadu_si128(v8 + 1);
      }
      else
      {
        v50 = (__m128i *)v8->m128i_i64[0];
        v52.m128i_i64[0] = v8[1].m128i_i64[0];
      }
      v51 = v8->m128i_i64[1];
      v8->m128i_i64[0] = (__int64)v8[1].m128i_i64;
      v8->m128i_i64[1] = 0;
      v8[1].m128i_i8[0] = 0;
      v9 = 15;
      v10 = 15;
      if ( v50 != &v52 )
        v10 = v52.m128i_i64[0];
      v11 = v51 + v54;
      if ( v51 + v54 <= v10 )
        goto LABEL_12;
      if ( v53 != v55 )
        v9 = v55[0];
      if ( v11 <= v9 )
      {
        v12 = (__m128i *)sub_2241130(&v53, 0, 0, v50, v51);
        v56 = &v58;
        v13 = (__m128i *)v12->m128i_i64[0];
        v14 = v12 + 1;
        if ( (__m128i *)v12->m128i_i64[0] != &v12[1] )
          goto LABEL_13;
      }
      else
      {
LABEL_12:
        v12 = (__m128i *)sub_2241490(&v50, v53, v54, v11);
        v56 = &v58;
        v13 = (__m128i *)v12->m128i_i64[0];
        v14 = v12 + 1;
        if ( (__m128i *)v12->m128i_i64[0] != &v12[1] )
        {
LABEL_13:
          v56 = v13;
          v58.m128i_i64[0] = v12[1].m128i_i64[0];
          goto LABEL_14;
        }
      }
LABEL_114:
      v58 = _mm_loadu_si128(v12 + 1);
LABEL_14:
      v57 = v12->m128i_i64[1];
      v15 = v57;
      v12->m128i_i64[0] = (__int64)v14;
      v12->m128i_i64[1] = 0;
      v12[1].m128i_i8[0] = 0;
      sub_2241490(a1, v56, v57, v15);
      if ( v56 != &v58 )
        j_j___libc_free_0(v56, v58.m128i_i64[0] + 1);
      if ( v50 != &v52 )
        j_j___libc_free_0(v50, v52.m128i_i64[0] + 1);
      if ( (_QWORD *)v48[0] != v49 )
        j_j___libc_free_0(v48[0], v49[0] + 1LL);
LABEL_20:
      if ( v53 != v55 )
        j_j___libc_free_0(v53, v55[0] + 1LL);
      return a1;
    case 14:
      sub_15DF750(&v53, *(_QWORD *)(a2 + 24));
      v18 = *(_QWORD *)(a2 + 32);
      if ( v18 )
      {
        v19 = &v58.m128i_i8[5];
        do
        {
          *--v19 = v18 % 0xA + 48;
          v24 = v18;
          v18 /= 0xAu;
        }
        while ( v24 > 9 );
      }
      else
      {
        v58.m128i_i8[4] = 48;
        v19 = &v58.m128i_i8[4];
      }
      v48[0] = (__int64)v49;
      sub_15DE6F0(v48, v19, (__int64)v58.m128i_i64 + 5);
      v20 = (__m128i *)sub_2241130(v48, 0, 0, "a", 1);
      v50 = &v52;
      if ( (__m128i *)v20->m128i_i64[0] == &v20[1] )
      {
        v52 = _mm_loadu_si128(v20 + 1);
      }
      else
      {
        v50 = (__m128i *)v20->m128i_i64[0];
        v52.m128i_i64[0] = v20[1].m128i_i64[0];
      }
      v51 = v20->m128i_i64[1];
      v20->m128i_i64[0] = (__int64)v20[1].m128i_i64;
      v20->m128i_i64[1] = 0;
      v20[1].m128i_i8[0] = 0;
      v21 = 15;
      v22 = 15;
      if ( v50 != &v52 )
        v22 = v52.m128i_i64[0];
      v23 = v51 + v54;
      if ( v51 + v54 <= v22 )
        goto LABEL_37;
      if ( v53 != v55 )
        v21 = v55[0];
      if ( v23 <= v21 )
      {
        v12 = (__m128i *)sub_2241130(&v53, 0, 0, v50, v51);
        v56 = &v58;
        v13 = (__m128i *)v12->m128i_i64[0];
        v14 = v12 + 1;
        if ( (__m128i *)v12->m128i_i64[0] != &v12[1] )
          goto LABEL_13;
      }
      else
      {
LABEL_37:
        v12 = (__m128i *)sub_2241490(&v50, v53, v54, v23);
        v56 = &v58;
        v13 = (__m128i *)v12->m128i_i64[0];
        v14 = v12 + 1;
        if ( (__m128i *)v12->m128i_i64[0] != &v12[1] )
          goto LABEL_13;
      }
      goto LABEL_114;
    case 13:
      if ( (*(_BYTE *)(a2 + 9) & 4) != 0 )
      {
        sub_2241490(a1, &unk_3F2CC08, 3, a4);
        v34 = *(_QWORD **)(a2 + 16);
        for ( i = &v34[*(unsigned int *)(a2 + 12)]; i != v34; ++v34 )
        {
          sub_15DF750(&v56, *v34);
          sub_2241490(a1, v56, v57, v36);
          if ( v56 != &v58 )
            j_j___libc_free_0(v56, v58.m128i_i64[0] + 1);
        }
      }
      else
      {
        sub_2241490(a1, "s_", 2, a4);
        v27 = sub_1643640(a2);
        if ( v25 > 0x3FFFFFFFFFFFFFFFLL - *(_QWORD *)(a1 + 8) )
          goto LABEL_116;
        sub_2241490(a1, v27, v25, v26);
      }
      if ( *(_QWORD *)(a1 + 8) != 0x3FFFFFFFFFFFFFFFLL )
      {
        sub_2241490(a1, "s", 1, v28);
        return a1;
      }
      goto LABEL_116;
    case 12:
      sub_15DF750(&v53, **(_QWORD **)(a2 + 16));
      v29 = (__m128i *)sub_2241130(&v53, 0, 0, "f_", 2);
      v56 = &v58;
      if ( (__m128i *)v29->m128i_i64[0] == &v29[1] )
      {
        v58 = _mm_loadu_si128(v29 + 1);
      }
      else
      {
        v56 = (__m128i *)v29->m128i_i64[0];
        v58.m128i_i64[0] = v29[1].m128i_i64[0];
      }
      v57 = v29->m128i_i64[1];
      v30 = v57;
      v29->m128i_i64[0] = (__int64)v29[1].m128i_i64;
      v29->m128i_i64[1] = 0;
      v29[1].m128i_i8[0] = 0;
      sub_2241490(a1, v56, v57, v30);
      if ( v56 != &v58 )
        j_j___libc_free_0(v56, v58.m128i_i64[0] + 1);
      if ( v53 != v55 )
        j_j___libc_free_0(v53, v55[0] + 1LL);
      v32 = 0;
      if ( *(_DWORD *)(a2 + 12) != 1 )
      {
        do
        {
          sub_15DF750(&v56, *(_QWORD *)(*(_QWORD *)(a2 + 16) + 8LL * (unsigned int)(v32 + 1)));
          sub_2241490(a1, v56, v57, v33);
          if ( v56 != &v58 )
            j_j___libc_free_0(v56, v58.m128i_i64[0] + 1);
          ++v32;
        }
        while ( (unsigned int)(*(_DWORD *)(a2 + 12) - 1) > v32 );
      }
      if ( *(_DWORD *)(a2 + 8) >> 8 )
      {
        if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - *(_QWORD *)(a1 + 8)) <= 5 )
          goto LABEL_116;
        sub_2241490(a1, "vararg", 6, v31);
      }
      if ( *(_QWORD *)(a1 + 8) != 0x3FFFFFFFFFFFFFFFLL )
      {
        sub_2241490(a1, "f", 1, v31);
        return a1;
      }
LABEL_116:
      sub_4262D8((__int64)"basic_string::append");
    case 16:
      sub_15DF750(&v53, **(_QWORD **)(a2 + 16));
      v41 = *(unsigned int *)(a2 + 32);
      if ( *(_DWORD *)(a2 + 32) )
      {
        v42 = &v58.m128i_i8[5];
        do
        {
          *--v42 = v41 % 0xA + 48;
          v46 = v41;
          v41 /= 0xAu;
        }
        while ( v46 > 9 );
      }
      else
      {
        v58.m128i_i8[4] = 48;
        v42 = &v58.m128i_i8[4];
      }
      v48[0] = (__int64)v49;
      sub_15DE6F0(v48, v42, (__int64)v58.m128i_i64 + 5);
      v43 = (__m128i *)sub_2241130(v48, 0, 0, "v", 1);
      v50 = &v52;
      if ( (__m128i *)v43->m128i_i64[0] == &v43[1] )
      {
        v52 = _mm_loadu_si128(v43 + 1);
      }
      else
      {
        v50 = (__m128i *)v43->m128i_i64[0];
        v52.m128i_i64[0] = v43[1].m128i_i64[0];
      }
      v51 = v43->m128i_i64[1];
      v43->m128i_i64[0] = (__int64)v43[1].m128i_i64;
      v43->m128i_i64[1] = 0;
      v43[1].m128i_i8[0] = 0;
      v44 = 15;
      v45 = 15;
      if ( v50 != &v52 )
        v45 = v52.m128i_i64[0];
      if ( v51 + v54 <= v45 )
        goto LABEL_104;
      if ( v53 != v55 )
        v44 = v55[0];
      if ( v51 + v54 <= v44 )
      {
        v12 = (__m128i *)sub_2241130(&v53, 0, 0, v50, v51);
        v56 = &v58;
        v13 = (__m128i *)v12->m128i_i64[0];
        v14 = v12 + 1;
        if ( (__m128i *)v12->m128i_i64[0] != &v12[1] )
          goto LABEL_13;
      }
      else
      {
LABEL_104:
        v12 = (__m128i *)sub_2241490(&v50, v53, v54, v50);
        v56 = &v58;
        v13 = (__m128i *)v12->m128i_i64[0];
        v14 = v12 + 1;
        if ( (__m128i *)v12->m128i_i64[0] != &v12[1] )
          goto LABEL_13;
      }
      goto LABEL_114;
  }
  switch ( *(_BYTE *)(a2 + 8) )
  {
    case 0:
      sub_2241490(a1, "isVoid", 6, a4);
      break;
    case 1:
      sub_2241490(a1, "f16", 3, a4);
      break;
    case 2:
      sub_2241490(a1, "f32", 3, a4);
      break;
    case 3:
      sub_2241490(a1, "f64", 3, a4);
      break;
    case 4:
      sub_2241490(a1, "f80", 3, a4);
      break;
    case 5:
      sub_2241490(a1, "f128", 4, a4);
      break;
    case 6:
      sub_2241490(a1, "ppcf128", 7, a4);
      break;
    case 7:
    case 0xA:
    case 0xB:
      v37 = *(_DWORD *)(a2 + 8) >> 8;
      if ( *(_DWORD *)(a2 + 8) >> 8 )
      {
        v38 = &v58.m128i_i8[5];
        do
        {
          *--v38 = v37 % 0xA + 48;
          v47 = v37;
          v37 /= 0xAu;
        }
        while ( v47 > 9 );
      }
      else
      {
        v58.m128i_i8[4] = 48;
        v38 = &v58.m128i_i8[4];
      }
      v53 = v55;
      sub_15DE6F0((__int64 *)&v53, v38, (__int64)v58.m128i_i64 + 5);
      v39 = (__m128i *)sub_2241130(&v53, 0, 0, "i", 1);
      v56 = &v58;
      if ( (__m128i *)v39->m128i_i64[0] == &v39[1] )
      {
        v58 = _mm_loadu_si128(v39 + 1);
      }
      else
      {
        v56 = (__m128i *)v39->m128i_i64[0];
        v58.m128i_i64[0] = v39[1].m128i_i64[0];
      }
      v57 = v39->m128i_i64[1];
      v40 = v57;
      v39->m128i_i64[0] = (__int64)v39[1].m128i_i64;
      v39->m128i_i64[1] = 0;
      v39[1].m128i_i8[0] = 0;
      sub_2241490(a1, v56, v57, v40);
      if ( v56 != &v58 )
        j_j___libc_free_0(v56, v58.m128i_i64[0] + 1);
      goto LABEL_20;
    case 8:
      sub_2241490(a1, "Metadata", 8, a4);
      break;
    case 9:
      sub_2241490(a1, "x86mmx", 6, a4);
      break;
  }
  return a1;
}
