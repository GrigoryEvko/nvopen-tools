// Function: sub_32204E0
// Address: 0x32204e0
//
__int64 __fastcall sub_32204E0(void **a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  void **v6; // r10
  const __m128i *v10; // r8
  __int64 v11; // r12
  void *v12; // rdx
  __int64 v13; // rax
  unsigned __int64 v14; // rdx
  unsigned __int64 v15; // r11
  __m128i *v16; // rax
  char v17; // al
  __int64 v18; // rbx
  __int64 v19; // rsi
  __int64 v20; // rax
  __int64 v21; // rbx
  char v22; // al
  unsigned __int64 *v23; // r14
  unsigned __int64 *v24; // rbx
  unsigned __int64 *v25; // r13
  __int64 v27; // rdi
  const void *v28; // rsi
  void **v30; // [rsp+10h] [rbp-100h]
  void **v31; // [rsp+10h] [rbp-100h]
  void **v32; // [rsp+18h] [rbp-F8h]
  __int64 v33; // [rsp+18h] [rbp-F8h]
  char *v34; // [rsp+18h] [rbp-F8h]
  _BYTE v35[32]; // [rsp+20h] [rbp-F0h] BYREF
  _QWORD v36[3]; // [rsp+40h] [rbp-D0h] BYREF
  char v37; // [rsp+58h] [rbp-B8h]
  void *v38; // [rsp+60h] [rbp-B0h] BYREF
  void *v39; // [rsp+68h] [rbp-A8h]
  __int64 v40; // [rsp+70h] [rbp-A0h]
  unsigned __int64 v41; // [rsp+78h] [rbp-98h]
  __int64 v42; // [rsp+80h] [rbp-90h]
  _BYTE v43[48]; // [rsp+88h] [rbp-88h] BYREF
  __int64 v44; // [rsp+B8h] [rbp-58h]
  unsigned __int64 v45; // [rsp+C0h] [rbp-50h]
  __int64 v46; // [rsp+C8h] [rbp-48h]
  _QWORD *v47; // [rsp+D0h] [rbp-40h]
  char v48; // [rsp+D8h] [rbp-38h]

  v6 = a1;
  v10 = (const __m128i *)&v38;
  v11 = *a3;
  v12 = *a1;
  v39 = a1[1];
  v38 = v12;
  v40 = *(_QWORD *)(v11 + 1192);
  v41 = (__int64)(*(_QWORD *)(v11 + 1472) - *(_QWORD *)(v11 + 1464)) >> 5;
  v13 = *(unsigned int *)(v11 + 152);
  v14 = *(_QWORD *)(v11 + 144);
  v15 = v13 + 1;
  if ( v13 + 1 > (unsigned __int64)*(unsigned int *)(v11 + 156) )
  {
    v27 = v11 + 144;
    v28 = (const void *)(v11 + 160);
    if ( v14 > (unsigned __int64)&v38 )
    {
      v31 = v6;
      sub_C8D5F0(v27, v28, v13 + 1, 0x20u, (__int64)&v38, a6);
      v14 = *(_QWORD *)(v11 + 144);
      v13 = *(unsigned int *)(v11 + 152);
      v10 = (const __m128i *)&v38;
      v6 = v31;
    }
    else
    {
      v30 = v6;
      if ( (unsigned __int64)&v38 >= v14 + 32 * v13 )
      {
        sub_C8D5F0(v27, v28, v15, 0x20u, (__int64)&v38, a6);
        v14 = *(_QWORD *)(v11 + 144);
        v13 = *(unsigned int *)(v11 + 152);
        v6 = v30;
        v10 = (const __m128i *)&v38;
      }
      else
      {
        v34 = (char *)&v38 - v14;
        sub_C8D5F0(v27, v28, v15, 0x20u, (__int64)&v38, a6);
        v14 = *(_QWORD *)(v11 + 144);
        v6 = v30;
        v10 = (const __m128i *)&v34[v14];
        v13 = *(unsigned int *)(v11 + 152);
      }
    }
  }
  v32 = v6;
  v16 = (__m128i *)(v14 + 32 * v13);
  *v16 = _mm_loadu_si128(v10);
  v16[1] = _mm_loadu_si128(v10 + 1);
  LOBYTE(v16) = *(_BYTE *)(v11 + 1496);
  ++*(_DWORD *)(v11 + 152);
  v36[0] = &unk_4A35738;
  v36[1] = v11 + 1184;
  v36[2] = v11 + 1464;
  v37 = (char)v16;
  v17 = sub_31DF670(a2);
  v48 = 0;
  v42 = 0x200000000LL;
  v40 = a5;
  v18 = (__int64)v32[2];
  LOBYTE(v39) = 0;
  v38 = &unk_4A35B90;
  v41 = (unsigned __int64)v43;
  v45 = v45 & 0xE00000000000LL | ((unsigned __int64)(v17 & 0xF) << 41);
  v44 = 0;
  v46 = 0;
  v47 = v36;
  v19 = *(_QWORD *)(*(_QWORD *)v18 + 16LL);
  sub_AF47B0((__int64)v35, (unsigned __int64 *)v19, *(unsigned __int64 **)(*(_QWORD *)v18 + 24LL));
  if ( v35[16] )
  {
    v20 = (__int64)v32[2];
    v21 = v20;
    v33 = v20 + 80LL * *((unsigned int *)v32 + 6);
    if ( v33 != v20 )
    {
      do
      {
        v19 = a4;
        sub_32200A0(a2, a4, v21, (unsigned __int64)&v38);
        v21 += 80;
      }
      while ( v33 != v21 );
    }
  }
  else
  {
    v19 = a4;
    sub_32200A0(a2, a4, v18, (unsigned __int64)&v38);
  }
  sub_3243D40(&v38);
  if ( HIBYTE(v45) )
  {
    v22 = BYTE6(v45);
    *((_BYTE *)a3 + 33) = 1;
    *((_BYTE *)a3 + 32) = v22;
  }
  v23 = (unsigned __int64 *)v46;
  v38 = &unk_4A35B90;
  if ( v46 )
  {
    v24 = *(unsigned __int64 **)(v46 + 64);
    v25 = *(unsigned __int64 **)(v46 + 56);
    if ( v24 != v25 )
    {
      do
      {
        if ( (unsigned __int64 *)*v25 != v25 + 2 )
          j_j___libc_free_0(*v25);
        v25 += 4;
      }
      while ( v24 != v25 );
      v25 = (unsigned __int64 *)v23[7];
    }
    if ( v25 )
      j_j___libc_free_0((unsigned __int64)v25);
    if ( (unsigned __int64 *)*v23 != v23 + 3 )
      _libc_free(*v23);
    v19 = 112;
    j_j___libc_free_0((unsigned __int64)v23);
  }
  if ( (_BYTE *)v41 != v43 )
    _libc_free(v41);
  return sub_372FBA0(v11, v19);
}
