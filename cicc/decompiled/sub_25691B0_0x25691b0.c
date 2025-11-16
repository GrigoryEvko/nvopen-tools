// Function: sub_25691B0
// Address: 0x25691b0
//
__int64 (__fastcall *__fastcall sub_25691B0(__int64 a1, __int64 a2))(__int64, __int64, __int64)
{
  __int64 v3; // rax
  unsigned __int64 *v4; // rdx
  unsigned __int64 *v5; // r12
  unsigned int v6; // esi
  __int64 v7; // r10
  __int64 v8; // r9
  __int64 *v9; // rax
  unsigned __int32 v10; // edx
  __int64 *v11; // rbx
  __int64 v12; // r8
  __int64 v13; // rdx
  __int64 v14; // r9
  int v15; // eax
  __int64 v16; // rbx
  unsigned __int64 *v17; // rdx
  unsigned __int64 *v18; // r12
  __int64 v19; // r13
  unsigned int v20; // esi
  __int64 v21; // rcx
  __int64 v22; // r11
  __int64 v23; // r10
  __int64 v24; // r9
  __int64 *v25; // rax
  unsigned __int32 i; // edx
  __int64 *v27; // rbx
  __int64 v28; // r8
  unsigned __int32 v29; // edx
  __int64 v30; // rdx
  __int64 *v31; // r15
  int v32; // eax
  __int64 v33; // rbx
  __m128i *v35; // rcx
  __int64 v36; // rax
  bool v37; // zf
  __m128i *v38; // rax
  __m128i *v39; // rdi
  unsigned __int64 v40; // rdi
  int v41; // r15d
  __m128i *v42; // rcx
  __int64 v43; // rax
  __int64 v44; // r9
  __m128i *v45; // rcx
  __int64 v46; // rax
  __m128i *v47; // rax
  __m128i *v48; // rdi
  unsigned __int64 v49; // rdi
  int v50; // eax
  __m128i *v51; // rcx
  int v52; // ecx
  __m128i v53; // xmm1
  int v54; // ecx
  __m128i v55; // xmm0
  int v56; // ecx
  int v57; // ecx
  __int64 v58; // [rsp+8h] [rbp-E8h]
  __int64 v60; // [rsp+18h] [rbp-D8h]
  __m128i *v61; // [rsp+20h] [rbp-D0h]
  __m128i *v62; // [rsp+20h] [rbp-D0h]
  int v63; // [rsp+28h] [rbp-C8h]
  int v64; // [rsp+28h] [rbp-C8h]
  __m128i *v65; // [rsp+28h] [rbp-C8h]
  __m128i *v66; // [rsp+30h] [rbp-C0h]
  int v67; // [rsp+30h] [rbp-C0h]
  __int64 v68; // [rsp+38h] [rbp-B8h]
  __int64 v69; // [rsp+38h] [rbp-B8h]
  __m128i *v70; // [rsp+38h] [rbp-B8h]
  unsigned __int64 *v71; // [rsp+48h] [rbp-A8h]
  unsigned __int64 *v72; // [rsp+48h] [rbp-A8h]
  char v73; // [rsp+5Fh] [rbp-91h] BYREF
  __int64 v74; // [rsp+60h] [rbp-90h] BYREF
  __int64 *v75; // [rsp+68h] [rbp-88h] BYREF
  __m128i v76; // [rsp+70h] [rbp-80h] BYREF
  _QWORD v77[4]; // [rsp+80h] [rbp-70h] BYREF
  __int64 v78; // [rsp+A0h] [rbp-50h] BYREF
  int v79; // [rsp+A8h] [rbp-48h]
  __int64 (__fastcall *v80)(__m128i *, __int64, int); // [rsp+B0h] [rbp-40h]
  __int64 (*v81)(); // [rsp+B8h] [rbp-38h]

  v3 = sub_25096F0((_QWORD *)(a1 + 72));
  v74 = sub_2555710(*(_QWORD *)(*(_QWORD *)(a2 + 208) + 240LL), v3, 0);
  v77[0] = &v74;
  v78 = 0xB00000005LL;
  v77[1] = a1;
  v77[2] = a2;
  v73 = 0;
  v79 = 56;
  sub_2526370(
    a2,
    (__int64 (__fastcall *)(__int64, unsigned __int64, __int64))sub_2574FF0,
    (__int64)v77,
    a1,
    (int *)&v78,
    3,
    &v73,
    0,
    1u);
  v4 = *(unsigned __int64 **)(a1 + 136);
  v81 = sub_2535080;
  v80 = (__int64 (__fastcall *)(__m128i *, __int64, int))sub_25350A0;
  v71 = &v4[2 * *(unsigned int *)(a1 + 144)];
  if ( v4 != v71 )
  {
    v5 = v4;
    v58 = a2 + 32;
    do
    {
      sub_250D230((unsigned __int64 *)&v76, *v5, 3, 0);
      v6 = *(_DWORD *)(a2 + 56);
      if ( v6 )
      {
        v63 = 1;
        v7 = *(_QWORD *)(a2 + 40);
        v8 = unk_4FEE4D0;
        v68 = qword_4FEE4D8;
        v9 = 0;
        v10 = (v6 - 1)
            & (((unsigned __int32)v76.m128i_i32[2] >> 9)
             ^ ((unsigned __int32)v76.m128i_i32[2] >> 4)
             ^ (16 * (((unsigned __int32)v76.m128i_i32[0] >> 9) ^ ((unsigned __int32)v76.m128i_i32[0] >> 4))));
        while ( 1 )
        {
          v11 = (__int64 *)(v7 + ((unsigned __int64)v10 << 6));
          v12 = *v11;
          if ( *(_OWORD *)&v76 == *(_OWORD *)v11 )
          {
            v13 = *((unsigned int *)v11 + 6);
            v14 = (__int64)(v11 + 2);
            v15 = v13;
            if ( *((_DWORD *)v11 + 7) > (unsigned int)v13 )
              goto LABEL_10;
            v43 = sub_C8D7D0((__int64)(v11 + 2), (__int64)(v11 + 4), 0, 0x20u, (unsigned __int64 *)&v75, v14);
            v44 = (__int64)(v11 + 2);
            v45 = (__m128i *)v43;
            v46 = 2LL * *((unsigned int *)v11 + 6);
            v37 = &v45[v46] == 0;
            v47 = &v45[v46];
            v48 = v47;
            if ( !v37 )
            {
              v47[1].m128i_i64[0] = 0;
              if ( v80 )
              {
                v62 = v45;
                v80(v47, (__int64)&v78, 2);
                v45 = v62;
                v44 = (__int64)(v11 + 2);
                v48[1].m128i_i64[1] = (__int64)v81;
                v48[1].m128i_i64[0] = (__int64)v80;
              }
            }
            v66 = v45;
            sub_255FA70(v44, v45);
            v49 = v11[2];
            v50 = (int)v75;
            v51 = v66;
            if ( v11 + 4 != (__int64 *)v49 )
            {
              v67 = (int)v75;
              v70 = v51;
              _libc_free(v49);
              v50 = v67;
              v51 = v70;
            }
            ++*((_DWORD *)v11 + 6);
            v11[2] = (__int64)v51;
            *((_DWORD *)v11 + 7) = v50;
            goto LABEL_15;
          }
          if ( unk_4FEE4D0 == v12 && qword_4FEE4D8 == v11[1] )
            break;
          if ( qword_4FEE4C0[0] == v12 && v11[1] == qword_4FEE4C0[1] && !v9 )
            v9 = (__int64 *)(v7 + ((unsigned __int64)v10 << 6));
          v10 = (v6 - 1) & (v63 + v10);
          ++v63;
        }
        v56 = *(_DWORD *)(a2 + 48);
        if ( !v9 )
          v9 = (__int64 *)(v7 + ((unsigned __int64)v10 << 6));
        ++*(_QWORD *)(a2 + 32);
        v54 = v56 + 1;
        v75 = v9;
        if ( 4 * v54 >= 3 * v6 )
          goto LABEL_53;
        if ( v6 - *(_DWORD *)(a2 + 52) - v54 > v6 >> 3 )
          goto LABEL_55;
      }
      else
      {
        ++*(_QWORD *)(a2 + 32);
        v75 = 0;
LABEL_53:
        v6 *= 2;
      }
      sub_2568D00(v58, v6);
      sub_255C130(v58, v76.m128i_i64, &v75);
      v8 = unk_4FEE4D0;
      v54 = *(_DWORD *)(a2 + 48) + 1;
      v68 = qword_4FEE4D8;
      v9 = v75;
LABEL_55:
      *(_DWORD *)(a2 + 48) = v54;
      if ( *v9 != v8 || v9[1] != v68 )
        --*(_DWORD *)(a2 + 52);
      v55 = _mm_loadu_si128(&v76);
      v14 = (__int64)(v9 + 2);
      v9[2] = (__int64)(v9 + 4);
      v13 = 0;
      v9[3] = 0x100000000LL;
      *(__m128i *)v9 = v55;
      v15 = 0;
LABEL_10:
      v16 = *(_QWORD *)v14 + 32 * v13;
      if ( v16 )
      {
        *(_QWORD *)(v16 + 16) = 0;
        if ( v80 )
        {
          v69 = v14;
          v80((__m128i *)v16, (__int64)&v78, 2);
          v14 = v69;
          *(_QWORD *)(v16 + 24) = v81;
          *(_QWORD *)(v16 + 16) = v80;
        }
        v15 = *(_DWORD *)(v14 + 8);
      }
      *(_DWORD *)(v14 + 8) = v15 + 1;
LABEL_15:
      v5 += 2;
    }
    while ( v71 != v5 );
  }
  v17 = *(unsigned __int64 **)(a1 + 184);
  v72 = &v17[2 * *(unsigned int *)(a1 + 192)];
  if ( v72 != v17 )
  {
    v18 = *(unsigned __int64 **)(a1 + 184);
    v19 = a2;
    v60 = a2 + 32;
    do
    {
      sub_250D230((unsigned __int64 *)&v76, *v18, 3, 0);
      v20 = *(_DWORD *)(v19 + 56);
      if ( v20 )
      {
        v21 = v76.m128i_i64[0];
        v64 = 1;
        v22 = *(_QWORD *)(v19 + 40);
        v23 = unk_4FEE4D0;
        v24 = qword_4FEE4D8;
        v25 = 0;
        for ( i = (v20 - 1)
                & (((unsigned __int32)v76.m128i_i32[2] >> 9)
                 ^ ((unsigned __int32)v76.m128i_i32[2] >> 4)
                 ^ (16 * (((unsigned __int32)v76.m128i_i32[0] >> 9) ^ ((unsigned __int32)v76.m128i_i32[0] >> 4))));
              ;
              i = (v20 - 1) & v29 )
        {
          v27 = (__int64 *)(v22 + ((unsigned __int64)i << 6));
          v28 = *v27;
          if ( *(_OWORD *)&v76 == *(_OWORD *)v27 )
          {
            v30 = *((unsigned int *)v27 + 6);
            v31 = v27 + 2;
            v32 = v30;
            if ( *((_DWORD *)v27 + 7) > (unsigned int)v30 )
              goto LABEL_25;
            v35 = (__m128i *)sub_C8D7D0(
                               (__int64)(v27 + 2),
                               (__int64)(v27 + 4),
                               0,
                               0x20u,
                               (unsigned __int64 *)&v75,
                               qword_4FEE4D8);
            v36 = 2LL * *((unsigned int *)v27 + 6);
            v37 = &v35[v36] == 0;
            v38 = &v35[v36];
            v39 = v38;
            if ( !v37 )
            {
              v38[1].m128i_i64[0] = 0;
              if ( v80 )
              {
                v61 = v35;
                v80(v38, (__int64)&v78, 2);
                v35 = v61;
                v39[1].m128i_i64[1] = (__int64)v81;
                v39[1].m128i_i64[0] = (__int64)v80;
              }
            }
            v65 = v35;
            sub_255FA70((__int64)(v27 + 2), v35);
            v40 = v27[2];
            v41 = (int)v75;
            v42 = v65;
            if ( v27 + 4 != (__int64 *)v40 )
            {
              _libc_free(v40);
              v42 = v65;
            }
            ++*((_DWORD *)v27 + 6);
            v27[2] = (__int64)v42;
            *((_DWORD *)v27 + 7) = v41;
            goto LABEL_30;
          }
          if ( unk_4FEE4D0 == v28 && qword_4FEE4D8 == v27[1] )
            break;
          if ( qword_4FEE4C0[0] == v28 && v27[1] == qword_4FEE4C0[1] && !v25 )
            v25 = (__int64 *)(v22 + ((unsigned __int64)i << 6));
          v29 = v64 + i;
          ++v64;
        }
        v57 = *(_DWORD *)(v19 + 48);
        if ( !v25 )
          v25 = (__int64 *)(v22 + ((unsigned __int64)i << 6));
        ++*(_QWORD *)(v19 + 32);
        v52 = v57 + 1;
        v75 = v25;
        if ( 4 * v52 >= 3 * v20 )
          goto LABEL_45;
        if ( v20 - *(_DWORD *)(v19 + 52) - v52 > v20 >> 3 )
          goto LABEL_47;
      }
      else
      {
        ++*(_QWORD *)(v19 + 32);
        v75 = 0;
LABEL_45:
        v20 *= 2;
      }
      sub_2568D00(v60, v20);
      sub_255C130(v60, v76.m128i_i64, &v75);
      v52 = *(_DWORD *)(v19 + 48) + 1;
      v23 = unk_4FEE4D0;
      v24 = qword_4FEE4D8;
      v25 = v75;
LABEL_47:
      *(_DWORD *)(v19 + 48) = v52;
      if ( v23 != *v25 || v24 != v25[1] )
        --*(_DWORD *)(v19 + 52);
      v53 = _mm_loadu_si128(&v76);
      v31 = v25 + 2;
      v21 = 0x100000000LL;
      v25[2] = (__int64)(v25 + 4);
      v30 = 0;
      v25[3] = 0x100000000LL;
      *(__m128i *)v25 = v53;
      v32 = 0;
LABEL_25:
      v33 = *v31 + 32 * v30;
      if ( v33 )
      {
        *(_QWORD *)(v33 + 16) = 0;
        if ( v80 )
        {
          ((void (__fastcall *)(__int64, __int64 *, __int64, __int64, __int64, __int64))v80)(
            v33,
            &v78,
            2,
            v21,
            v28,
            v24);
          *(_QWORD *)(v33 + 24) = v81;
          *(_QWORD *)(v33 + 16) = v80;
        }
        v32 = *((_DWORD *)v31 + 2);
      }
      *((_DWORD *)v31 + 2) = v32 + 1;
LABEL_30:
      v18 += 2;
    }
    while ( v72 != v18 );
  }
  return sub_A17130((__int64)&v78);
}
