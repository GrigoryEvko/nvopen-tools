// Function: sub_335C390
// Address: 0x335c390
//
void __fastcall sub_335C390(__int64 a1, __int64 a2, void *a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v9; // rdx
  const __m128i *v10; // rbx
  int v11; // ecx
  unsigned __int64 v12; // rsi
  unsigned __int64 v13; // r12
  __m128i *v14; // rdx
  const __m128i *v15; // r13
  unsigned __int64 v16; // rdx
  __int64 *v17; // r12
  int v18; // eax
  int v19; // edx
  int v20; // r9d
  int v21; // r13d
  int v22; // ebx
  __int64 v23; // rax
  __int64 v24; // r11
  int v25; // edx
  char *v26; // r8
  char *v27; // rax
  size_t v28; // r9
  _BYTE *v29; // rdi
  char *srca; // [rsp+8h] [rbp-108h]
  void *src; // [rsp+8h] [rbp-108h]
  size_t nb; // [rsp+10h] [rbp-100h]
  size_t n; // [rsp+10h] [rbp-100h]
  size_t na; // [rsp+10h] [rbp-100h]
  char *v35; // [rsp+18h] [rbp-F8h]
  __int64 v36; // [rsp+18h] [rbp-F8h]
  __int64 v37; // [rsp+18h] [rbp-F8h]
  unsigned __int64 v38; // [rsp+20h] [rbp-F0h]
  int v39; // [rsp+20h] [rbp-F0h]
  __int64 v40; // [rsp+20h] [rbp-F0h]
  _BYTE *v41; // [rsp+30h] [rbp-E0h] BYREF
  __int64 v42; // [rsp+38h] [rbp-D8h]
  _BYTE v43[16]; // [rsp+40h] [rbp-D0h] BYREF
  _BYTE *v44; // [rsp+50h] [rbp-C0h] BYREF
  __int64 v45; // [rsp+58h] [rbp-B8h]
  _BYTE v46[176]; // [rsp+60h] [rbp-B0h] BYREF

  v9 = *(unsigned int *)(a1 + 64);
  v10 = *(const __m128i **)(a1 + 40);
  v45 = 0x800000000LL;
  v11 = 0;
  v12 = 40 * v9;
  v13 = v9;
  v14 = (__m128i *)v46;
  v15 = (const __m128i *)((char *)v10 + v12);
  v44 = v46;
  if ( v12 > 0x140 )
  {
    src = a3;
    n = a6;
    v36 = a5;
    sub_C8D5F0((__int64)&v44, v46, v13, 0x10u, a5, a6);
    v11 = v45;
    a3 = src;
    a6 = n;
    a5 = v36;
    v14 = (__m128i *)&v44[16 * (unsigned int)v45];
  }
  if ( v10 != v15 )
  {
    do
    {
      if ( v14 )
        *v14 = _mm_loadu_si128(v10);
      v10 = (const __m128i *)((char *)v10 + 40);
      ++v14;
    }
    while ( v15 != v10 );
    v11 = v45;
  }
  LODWORD(v13) = v11 + v13;
  LODWORD(v45) = v13;
  if ( a5 )
  {
    v13 = (unsigned int)v13;
    v16 = (unsigned int)v13 + 1LL;
    if ( v16 > HIDWORD(v45) )
    {
      na = (size_t)a3;
      v37 = a6;
      v40 = a5;
      sub_C8D5F0((__int64)&v44, v46, v16, 0x10u, a5, a6);
      v13 = (unsigned int)v45;
      a3 = (void *)na;
      a6 = v37;
      a5 = v40;
    }
    v17 = (__int64 *)&v44[16 * v13];
    *v17 = a5;
    v17[1] = a6;
    LODWORD(v45) = v45 + 1;
  }
  v18 = sub_33E5830(a2, a3);
  v20 = *(_DWORD *)(a1 + 24);
  v21 = v18;
  v22 = v19;
  if ( v20 >= 0 )
  {
    v42 = 0x200000000LL;
    v41 = v43;
    sub_33EC480(a2, a1, v20, v18, v19, v20, (__int64)v44, (unsigned int)v45);
  }
  else
  {
    v42 = 0x200000000LL;
    v23 = *(int *)(a1 + 104);
    v41 = v43;
    if ( (_DWORD)v23 )
    {
      if ( (_DWORD)v23 == 1 )
      {
        v27 = (char *)(a1 + 104);
        v26 = (char *)(a1 + 96);
      }
      else
      {
        v26 = (char *)(*(_QWORD *)(a1 + 96) & 0xFFFFFFFFFFFFFFF8LL);
        v27 = &v26[8 * v23];
      }
      v28 = v27 - v26;
      v24 = (v27 - v26) >> 3;
      if ( (unsigned __int64)(v27 - v26) <= 0x10 )
      {
        v29 = v43;
        v25 = 0;
      }
      else
      {
        srca = v26;
        nb = v27 - v26;
        v35 = v27;
        v38 = (v27 - v26) >> 3;
        sub_C8D5F0((__int64)&v41, v43, v38, 8u, (__int64)v26, v28);
        v25 = v42;
        LODWORD(v24) = v38;
        v27 = v35;
        v28 = nb;
        v26 = srca;
        v29 = &v41[8 * (unsigned int)v42];
      }
      if ( v26 == v27 )
      {
        v20 = *(_DWORD *)(a1 + 24);
      }
      else
      {
        v39 = v24;
        memcpy(v29, v26, v28);
        v25 = v42;
        v20 = *(_DWORD *)(a1 + 24);
        LODWORD(v24) = v39;
      }
    }
    else
    {
      LODWORD(v24) = 0;
      v25 = 0;
    }
    LODWORD(v42) = v24 + v25;
    sub_33EC480(a2, a1, v20, v21, v22, v20, (__int64)v44, (unsigned int)v45);
    sub_33E4DA0(a2, a1, v41, (unsigned int)v42);
  }
  if ( v41 != v43 )
    _libc_free((unsigned __int64)v41);
  if ( v44 != v46 )
    _libc_free((unsigned __int64)v44);
}
