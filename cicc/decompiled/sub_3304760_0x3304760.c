// Function: sub_3304760
// Address: 0x3304760
//
void __fastcall sub_3304760(__int64 *a1, __int64 a2, __int64 a3, int a4, __int64 a5, __int64 a6, int a7)
{
  int v7; // ebx
  __int64 v8; // r15
  __int64 v11; // rsi
  __int64 v12; // rdx
  __int64 *v13; // r13
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // r9
  unsigned __int64 v18; // rdx
  __int64 *m128i_i64; // rax
  __int64 v20; // rbx
  __int64 v21; // rbx
  __m128i v22; // xmm0
  __int64 v23; // rax
  unsigned __int64 v24; // rdx
  __int64 v25; // rdi
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 *v29; // r15
  __int64 v30; // rbx
  __int64 v31; // r13
  _QWORD *v32; // rax
  __int64 v33; // r10
  __int64 v34; // r11
  __int64 v35; // rax
  unsigned __int64 v36; // rdx
  __m128i *v37; // rax
  __int128 v38; // [rsp-10h] [rbp-130h]
  __int128 v39; // [rsp-10h] [rbp-130h]
  __int64 v41; // [rsp+10h] [rbp-110h]
  __int64 v42; // [rsp+18h] [rbp-108h]
  __int64 v43; // [rsp+30h] [rbp-F0h]
  __m128i v45; // [rsp+40h] [rbp-E0h] BYREF
  int v46; // [rsp+54h] [rbp-CCh]
  __int64 v47; // [rsp+58h] [rbp-C8h]
  __m128i *v48; // [rsp+60h] [rbp-C0h]
  __int64 v49; // [rsp+68h] [rbp-B8h]
  __m128i *v50; // [rsp+70h] [rbp-B0h]
  __int64 *v51; // [rsp+78h] [rbp-A8h]
  __int64 v52; // [rsp+80h] [rbp-A0h] BYREF
  int v53; // [rsp+88h] [rbp-98h]
  __int64 v54[2]; // [rsp+90h] [rbp-90h] BYREF
  __m128i *v55; // [rsp+A0h] [rbp-80h] BYREF
  __int64 v56; // [rsp+A8h] [rbp-78h]
  char v57; // [rsp+B0h] [rbp-70h] BYREF

  v11 = *(_QWORD *)(a5 + 80);
  v46 = a4;
  v47 = a5;
  v52 = v11;
  if ( v11 )
    sub_B96E90((__int64)&v52, v11, 1);
  v12 = *(unsigned int *)(a2 + 8);
  v53 = *(_DWORD *)(v47 + 72);
  v43 = *(_QWORD *)a2 + 8 * v12;
  if ( *(_QWORD *)a2 != v43 )
  {
    v51 = *(__int64 **)a2;
    v13 = &v52;
    WORD1(a2) = HIWORD(v7);
    v50 = (__m128i *)&v57;
    do
    {
      v17 = 0;
      v45.m128i_i64[0] = v8;
      v29 = v13;
      v30 = *v51;
      v55 = v50;
      v31 = v30;
      v20 = 0;
      v56 = 0x400000000LL;
      do
      {
        v32 = (_QWORD *)(v20 + *(_QWORD *)(v31 + 40));
        v33 = *v32;
        v34 = v32[1];
        if ( *((_DWORD *)v32 + 2) == v46 && *v32 == a3 )
        {
          v35 = (unsigned int)v56;
          v36 = (unsigned int)v56 + 1LL;
          if ( v36 > HIDWORD(v56) )
          {
            sub_C8D5F0((__int64)&v55, v50, v36, 0x10u, a5, v17);
            v35 = (unsigned int)v56;
          }
          v37 = &v55[v35];
          v37->m128i_i64[0] = v47;
          v37->m128i_i64[1] = a6;
          LODWORD(v56) = v56 + 1;
        }
        else
        {
          v14 = *(_QWORD *)(v47 + 48);
          LOWORD(a2) = *(_WORD *)v14;
          *((_QWORD *)&v38 + 1) = v34;
          *(_QWORD *)&v38 = v33;
          a5 = sub_33FAF80(*a1, a7, (_DWORD)v29, a2, *(_QWORD *)(v14 + 8), v17, v38);
          v15 = (unsigned int)v56;
          v17 = v16;
          v18 = (unsigned int)v56 + 1LL;
          if ( v18 > HIDWORD(v56) )
          {
            v41 = a5;
            v42 = v17;
            sub_C8D5F0((__int64)&v55, v50, v18, 0x10u, a5, v17);
            v15 = (unsigned int)v56;
            a5 = v41;
            v17 = v42;
          }
          m128i_i64 = v55[v15].m128i_i64;
          *m128i_i64 = a5;
          m128i_i64[1] = v17;
          LODWORD(v56) = v56 + 1;
        }
        v20 += 40;
      }
      while ( v20 != 80 );
      v21 = v31;
      v13 = v29;
      v8 = v45.m128i_i64[0];
      v22 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(v21 + 40) + 80LL));
      v23 = (unsigned int)v56;
      v24 = (unsigned int)v56 + 1LL;
      if ( v24 > HIDWORD(v56) )
      {
        v45 = v22;
        sub_C8D5F0((__int64)&v55, v50, v24, 0x10u, a5, v17);
        v23 = (unsigned int)v56;
        v22 = _mm_load_si128(&v45);
      }
      v55[v23] = v22;
      v25 = *a1;
      v48 = v55;
      LODWORD(v56) = v56 + 1;
      v49 = (unsigned int)v56;
      v26 = *(_QWORD *)(v21 + 48);
      LOWORD(v8) = *(_WORD *)v26;
      *((_QWORD *)&v39 + 1) = (unsigned int)v56;
      *(_QWORD *)&v39 = v55;
      v27 = sub_33FC220(v25, 208, (_DWORD)v13, v8, *(_QWORD *)(v26 + 8), v17, v39);
      v54[1] = v28;
      v54[0] = v27;
      sub_32EB790((__int64)a1, v21, v54, 1, 1);
      if ( v55 != v50 )
        _libc_free((unsigned __int64)v55);
      ++v51;
    }
    while ( (__int64 *)v43 != v51 );
  }
  if ( v52 )
    sub_B91220((__int64)&v52, v52);
}
