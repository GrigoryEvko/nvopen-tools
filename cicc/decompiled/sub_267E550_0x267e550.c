// Function: sub_267E550
// Address: 0x267e550
//
void __fastcall sub_267E550(__int64 a1, __m128i *a2, __int64 a3)
{
  unsigned int v6; // esi
  __int64 v7; // r11
  __int64 v8; // rdi
  __int64 v9; // r9
  __int64 v10; // rcx
  __int64 *v11; // rax
  unsigned int i; // edx
  __int64 *v13; // r12
  __int64 v14; // r8
  unsigned int v15; // edx
  __int64 v16; // rax
  __int64 *v17; // r13
  int v18; // edx
  __int64 v19; // rax
  bool v20; // zf
  __int64 v21; // rax
  __int64 v22; // rbx
  void (__fastcall *v23)(__int64, __int64, __int64, __int64, __int64, __int64); // rax
  __m128i *v24; // r15
  __int64 v25; // rax
  __m128i *v26; // rax
  __m128i *v27; // rcx
  void (__fastcall *v28)(__m128i *, __int64, __int64); // rax
  unsigned __int64 v29; // rdi
  int v30; // r13d
  int v31; // ecx
  int v32; // ecx
  __m128i v33; // xmm0
  int v34; // [rsp+14h] [rbp-5Ch]
  __int64 v35; // [rsp+20h] [rbp-50h]
  __int64 v36; // [rsp+28h] [rbp-48h]
  __m128i *v37; // [rsp+28h] [rbp-48h]
  __int64 *v38[7]; // [rsp+38h] [rbp-38h] BYREF

  v6 = *(_DWORD *)(a1 + 56);
  v36 = a1 + 32;
  if ( v6 )
  {
    v7 = *(_QWORD *)(a1 + 40);
    v8 = a2->m128i_i64[1];
    v34 = 1;
    v9 = qword_4FEE4D0;
    v35 = qword_4FEE4D8;
    v10 = a2->m128i_i64[0];
    v11 = 0;
    for ( i = (v6 - 1)
            & (((unsigned int)v8 >> 9)
             ^ ((unsigned int)v8 >> 4)
             ^ (16 * (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4)))); ; i = (v6 - 1) & v15 )
    {
      v13 = (__int64 *)(v7 + ((unsigned __int64)i << 6));
      v14 = *v13;
      if ( v10 == *v13 && v8 == v13[1] )
      {
        v16 = *((unsigned int *)v13 + 6);
        v17 = v13 + 2;
        v18 = v16;
        if ( *((_DWORD *)v13 + 7) > (unsigned int)v16 )
          goto LABEL_9;
        v24 = (__m128i *)sub_C8D7D0(
                           (__int64)(v13 + 2),
                           (__int64)(v13 + 4),
                           0,
                           0x20u,
                           (unsigned __int64 *)v38,
                           qword_4FEE4D0);
        v25 = 2LL * *((unsigned int *)v13 + 6);
        v20 = &v24[v25] == 0;
        v26 = &v24[v25];
        v27 = v26;
        if ( !v20 )
        {
          v26[1].m128i_i64[0] = 0;
          v28 = *(void (__fastcall **)(__m128i *, __int64, __int64))(a3 + 16);
          if ( v28 )
          {
            v37 = v27;
            v28(v27, a3, 2);
            v37[1].m128i_i64[1] = *(_QWORD *)(a3 + 24);
            v37[1].m128i_i64[0] = *(_QWORD *)(a3 + 16);
          }
        }
        sub_255FA70((__int64)(v13 + 2), v24);
        v29 = v13[2];
        v30 = (int)v38[0];
        if ( v13 + 4 != (__int64 *)v29 )
          _libc_free(v29);
        ++*((_DWORD *)v13 + 6);
        v13[2] = (__int64)v24;
        *((_DWORD *)v13 + 7) = v30;
        return;
      }
      if ( qword_4FEE4D0 == v14 && qword_4FEE4D8 == v13[1] )
        break;
      if ( qword_4FEE4C0[0] == v14 && v13[1] == qword_4FEE4C0[1] && !v11 )
        v11 = (__int64 *)(v7 + ((unsigned __int64)i << 6));
      v15 = v34 + i;
      ++v34;
    }
    v31 = *(_DWORD *)(a1 + 48);
    if ( !v11 )
      v11 = (__int64 *)(v7 + ((unsigned __int64)i << 6));
    ++*(_QWORD *)(a1 + 32);
    v32 = v31 + 1;
    v38[0] = v11;
    if ( 4 * v32 >= 3 * v6 )
      goto LABEL_35;
    if ( v6 - *(_DWORD *)(a1 + 52) - v32 <= v6 >> 3 )
      goto LABEL_36;
  }
  else
  {
    ++*(_QWORD *)(a1 + 32);
    v38[0] = 0;
LABEL_35:
    v6 *= 2;
LABEL_36:
    sub_2568D00(v36, v6);
    sub_255C130(v36, a2->m128i_i64, v38);
    v9 = qword_4FEE4D0;
    v32 = *(_DWORD *)(a1 + 48) + 1;
    v35 = qword_4FEE4D8;
    v11 = v38[0];
  }
  *(_DWORD *)(a1 + 48) = v32;
  if ( *v11 != v9 || v11[1] != v35 )
    --*(_DWORD *)(a1 + 52);
  v33 = _mm_loadu_si128(a2);
  v10 = 0x100000000LL;
  v17 = v11 + 2;
  v11[2] = (__int64)(v11 + 4);
  v18 = 0;
  v11[3] = 0x100000000LL;
  *(__m128i *)v11 = v33;
  v16 = 0;
LABEL_9:
  v19 = 32 * v16;
  v20 = *v17 + v19 == 0;
  v21 = *v17 + v19;
  v22 = v21;
  if ( !v20 )
  {
    *(_QWORD *)(v21 + 16) = 0;
    v23 = *(void (__fastcall **)(__int64, __int64, __int64, __int64, __int64, __int64))(a3 + 16);
    if ( v23 )
    {
      v23(v22, a3, 2, v10, v14, v9);
      *(_QWORD *)(v22 + 24) = *(_QWORD *)(a3 + 24);
      *(_QWORD *)(v22 + 16) = *(_QWORD *)(a3 + 16);
    }
    v18 = *((_DWORD *)v17 + 2);
  }
  *((_DWORD *)v17 + 2) = v18 + 1;
}
