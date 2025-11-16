// Function: sub_1BBE790
// Address: 0x1bbe790
//
void __fastcall sub_1BBE790(
        __int64 a1,
        __m128 a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9)
{
  __int64 v10; // rdi
  __int64 v11; // rsi
  __int64 v12; // rax
  __int64 v13; // r13
  unsigned __int64 v14; // rdi
  __int64 v15; // rax
  __int64 v16; // rbx
  __int64 v17; // r12
  __int64 *v18; // r12
  __int64 v19; // r14
  __int64 v20; // rbx
  unsigned __int64 v21; // rdi
  __int64 v22; // rdi
  __int64 v23; // rdi
  __int64 v24; // rdx
  __int64 v25; // rcx
  double v26; // xmm4_8
  double v27; // xmm5_8
  unsigned __int64 v28; // rdi
  unsigned __int64 v29; // rdi
  __int64 v30; // rbx
  unsigned __int64 v31; // r12
  __int64 v32; // rdi
  unsigned __int64 v33; // rdi
  unsigned __int64 *v34; // rbx
  unsigned __int64 *v35; // r12
  unsigned __int64 v36; // rdi
  unsigned __int64 v37; // rdi
  unsigned __int64 *v38; // rbx
  unsigned __int64 *v39; // r12
  __int64 v40; // [rsp+8h] [rbp-48h]
  __int64 v41; // [rsp+10h] [rbp-40h]
  __int64 *v42; // [rsp+18h] [rbp-38h]

  v10 = *(_QWORD *)(a1 + 1504);
  if ( v10 )
    j_j___libc_free_0(v10, *(_QWORD *)(a1 + 1520) - v10);
  j___libc_free_0(*(_QWORD *)(a1 + 1480));
  v11 = *(_QWORD *)(a1 + 1400);
  if ( v11 )
    sub_161E7C0(a1 + 1400, v11);
  v12 = *(unsigned int *)(a1 + 1288);
  if ( (_DWORD)v12 )
  {
    v38 = *(unsigned __int64 **)(a1 + 1272);
    v39 = &v38[5 * v12];
    do
    {
      if ( (unsigned __int64 *)*v38 != v38 + 2 )
        _libc_free(*v38);
      v38 += 5;
    }
    while ( v39 != v38 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 1272));
  v40 = *(_QWORD *)(a1 + 1232);
  v41 = *(_QWORD *)(a1 + 1224);
  if ( v40 != v41 )
  {
    do
    {
      v13 = *(_QWORD *)(v41 + 8);
      if ( v13 )
      {
        v14 = *(_QWORD *)(v13 + 104);
        if ( v14 != v13 + 120 )
          _libc_free(v14);
        v15 = *(unsigned int *)(v13 + 96);
        if ( (_DWORD)v15 )
        {
          v16 = *(_QWORD *)(v13 + 80);
          v17 = v16 + 88 * v15;
          do
          {
            if ( *(_QWORD *)v16 != -16 && *(_QWORD *)v16 != -8 && (*(_BYTE *)(v16 + 16) & 1) == 0 )
              j___libc_free_0(*(_QWORD *)(v16 + 24));
            v16 += 88;
          }
          while ( v17 != v16 );
        }
        j___libc_free_0(*(_QWORD *)(v13 + 80));
        j___libc_free_0(*(_QWORD *)(v13 + 48));
        v18 = *(__int64 **)(v13 + 8);
        v42 = *(__int64 **)(v13 + 16);
        if ( v42 != v18 )
        {
          do
          {
            v19 = *v18;
            if ( *v18 )
            {
              v20 = v19 + 112LL * *(_QWORD *)(v19 - 8);
              while ( v19 != v20 )
              {
                v20 -= 112;
                v21 = *(_QWORD *)(v20 + 32);
                if ( v21 != v20 + 48 )
                  _libc_free(v21);
              }
              j_j_j___libc_free_0_0(v19 - 8);
            }
            ++v18;
          }
          while ( v42 != v18 );
          v18 = *(__int64 **)(v13 + 8);
        }
        if ( v18 )
          j_j___libc_free_0(v18, *(_QWORD *)(v13 + 24) - (_QWORD)v18);
        v11 = 232;
        j_j___libc_free_0(v13, 232);
      }
      v41 += 16;
    }
    while ( v40 != v41 );
    v41 = *(_QWORD *)(a1 + 1224);
  }
  if ( v41 )
  {
    v11 = *(_QWORD *)(a1 + 1240) - v41;
    j_j___libc_free_0(v41, v11);
  }
  j___libc_free_0(*(_QWORD *)(a1 + 1200));
  v22 = *(_QWORD *)(a1 + 1168);
  if ( v22 )
  {
    v11 = *(_QWORD *)(a1 + 1184) - v22;
    j_j___libc_free_0(v22, v11);
  }
  j___libc_free_0(*(_QWORD *)(a1 + 1144));
  v23 = *(_QWORD *)(a1 + 1112);
  if ( v23 )
  {
    v11 = *(_QWORD *)(a1 + 1128) - v23;
    j_j___libc_free_0(v23, v11);
  }
  j___libc_free_0(*(_QWORD *)(a1 + 1088));
  v28 = *(_QWORD *)(a1 + 800);
  if ( v28 != *(_QWORD *)(a1 + 792) )
    _libc_free(v28);
  v29 = *(_QWORD *)(a1 + 384);
  if ( v29 != a1 + 400 )
    _libc_free(v29);
  v30 = *(_QWORD *)(a1 + 304);
  v31 = v30 + 8LL * *(unsigned int *)(a1 + 312);
  if ( v30 != v31 )
  {
    do
    {
      v32 = *(_QWORD *)(v31 - 8);
      v31 -= 8LL;
      if ( v32 )
        sub_164BEC0(v32, v11, v24, v25, a2, a3, a4, a5, v26, v27, a8, a9);
    }
    while ( v30 != v31 );
    v31 = *(_QWORD *)(a1 + 304);
  }
  if ( v31 != a1 + 320 )
    _libc_free(v31);
  j___libc_free_0(*(_QWORD *)(a1 + 280));
  v33 = *(_QWORD *)(a1 + 120);
  if ( v33 != *(_QWORD *)(a1 + 112) )
    _libc_free(v33);
  if ( (*(_BYTE *)(a1 + 32) & 1) == 0 )
    j___libc_free_0(*(_QWORD *)(a1 + 40));
  v34 = *(unsigned __int64 **)(a1 + 8);
  v35 = *(unsigned __int64 **)a1;
  if ( v34 != *(unsigned __int64 **)a1 )
  {
    do
    {
      v36 = v35[19];
      if ( (unsigned __int64 *)v36 != v35 + 21 )
        _libc_free(v36);
      v37 = v35[12];
      if ( (unsigned __int64 *)v37 != v35 + 14 )
        _libc_free(v37);
      if ( (unsigned __int64 *)*v35 != v35 + 2 )
        _libc_free(*v35);
      v35 += 22;
    }
    while ( v34 != v35 );
    v35 = *(unsigned __int64 **)a1;
  }
  if ( v35 )
    j_j___libc_free_0(v35, *(_QWORD *)(a1 + 16) - (_QWORD)v35);
}
