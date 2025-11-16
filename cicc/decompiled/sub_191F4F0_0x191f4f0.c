// Function: sub_191F4F0
// Address: 0x191f4f0
//
__int64 __fastcall sub_191F4F0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128 a7,
        __m128 a8,
        double a9,
        double a10,
        double a11,
        double a12,
        double a13,
        __m128 a14,
        __int64 a15,
        __int64 a16,
        __int64 a17,
        __int64 a18)
{
  __int64 *v20; // rdx
  bool v21; // zf
  __int64 v22; // r14
  unsigned int i; // ebx
  __int64 v24; // rdi
  __int64 v25; // rsi
  unsigned int v26; // r14d
  __int64 v27; // rdx
  int v28; // eax
  __int64 v29; // rdx
  _QWORD *v30; // rax
  _QWORD *j; // rdx
  __int64 v32; // rax
  _QWORD *v33; // rbx
  _QWORD *v34; // r12
  __int64 v35; // r13
  unsigned int v37; // ecx
  _QWORD *v38; // rdi
  unsigned int v39; // eax
  int v40; // eax
  unsigned __int64 v41; // rax
  unsigned __int64 v42; // rax
  int v43; // ebx
  __int64 v44; // r13
  _QWORD *v45; // rax
  __int64 v46; // rdx
  _QWORD *k; // rdx
  unsigned int v48; // eax
  unsigned int v49; // eax
  _QWORD *v50; // rax
  __int64 v51; // [rsp+10h] [rbp-60h] BYREF
  _QWORD *v52; // [rsp+18h] [rbp-58h]
  __int64 v53; // [rsp+20h] [rbp-50h]
  __int64 v54; // [rsp+28h] [rbp-48h]
  __int64 v55; // [rsp+30h] [rbp-40h]

  *(_QWORD *)(a1 + 40) = a3;
  *(_QWORD *)(a1 + 24) = a4;
  *(_QWORD *)a1 = a16;
  *(_QWORD *)(a1 + 344) = a16;
  *(_QWORD *)(a1 + 8) = a15;
  v20 = &v51;
  *(_QWORD *)(a1 + 32) = a5;
  *(_QWORD *)(a1 + 16) = a17;
  *(_QWORD *)(a1 + 144) = &v51;
  *(_QWORD *)(a1 + 104) = a18;
  v21 = byte_4FAE960 == 0;
  *(_QWORD *)(a1 + 352) = a4;
  *(_QWORD *)(a1 + 336) = a6;
  *(_BYTE *)(a1 + 368) = 1;
  v51 = 0;
  v52 = 0;
  v53 = 0;
  v54 = 0;
  v55 = a4;
  *(_BYTE *)(a1 + 784) = 1;
  if ( !v21 )
    sub_19141C0(a1, a2);
  v22 = *(_QWORD *)(a2 + 80);
  for ( i = 0; a2 + 72 != v22; i |= sub_1AA7EA0(v24, *(_QWORD *)(a1 + 24), a17, *(_QWORD *)a1, 0) )
  {
    v24 = v22 - 24;
    v22 = *(_QWORD *)(v22 + 8);
  }
  do
  {
    v25 = a2;
    v26 = i;
    i = sub_191F410(a1, a2, (__int64)v20, a4, a5, a6, a7, *(double *)a8.m128_u64, a9, a10, a11, a12, a13, a14);
  }
  while ( (_BYTE)i );
  if ( byte_4FAEF80 )
  {
    sub_1914110(a1);
    v49 = v26;
    do
    {
      v25 = a2;
      v26 = v49;
      v49 = sub_191C7F0(a1, a2, a7, a8, a9, a10, a11, a12, a13, a14);
    }
    while ( (_BYTE)v49 );
  }
  v27 = (unsigned int)dword_4FAECE0;
  if ( dword_4FAECE0 )
  {
    v25 = a2;
    v48 = sub_1913500((__int64 *)a1, a2, a7, *(double *)a8.m128_u64, a9, a10, a11, a12, a13, a14);
    if ( (_BYTE)v48 )
      v26 = v48;
  }
  sub_190DF50(a1, v25, v27, a4, a5, a6);
  v28 = *(_DWORD *)(a1 + 64);
  ++*(_QWORD *)(a1 + 48);
  if ( !v28 )
  {
    if ( !*(_DWORD *)(a1 + 68) )
      goto LABEL_14;
    v29 = *(unsigned int *)(a1 + 72);
    if ( (unsigned int)v29 > 0x40 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 56));
      *(_QWORD *)(a1 + 56) = 0;
      *(_QWORD *)(a1 + 64) = 0;
      *(_DWORD *)(a1 + 72) = 0;
      goto LABEL_14;
    }
    goto LABEL_11;
  }
  v37 = 4 * v28;
  v29 = *(unsigned int *)(a1 + 72);
  if ( (unsigned int)(4 * v28) < 0x40 )
    v37 = 64;
  if ( (unsigned int)v29 <= v37 )
  {
LABEL_11:
    v30 = *(_QWORD **)(a1 + 56);
    for ( j = &v30[v29]; j != v30; ++v30 )
      *v30 = -8;
    *(_QWORD *)(a1 + 64) = 0;
    goto LABEL_14;
  }
  v38 = *(_QWORD **)(a1 + 56);
  v39 = v28 - 1;
  if ( !v39 )
  {
    v44 = 1024;
    v43 = 128;
LABEL_34:
    j___libc_free_0(v38);
    *(_DWORD *)(a1 + 72) = v43;
    v45 = (_QWORD *)sub_22077B0(v44);
    v46 = *(unsigned int *)(a1 + 72);
    *(_QWORD *)(a1 + 64) = 0;
    *(_QWORD *)(a1 + 56) = v45;
    for ( k = &v45[v46]; k != v45; ++v45 )
    {
      if ( v45 )
        *v45 = -8;
    }
    goto LABEL_14;
  }
  _BitScanReverse(&v39, v39);
  v40 = 1 << (33 - (v39 ^ 0x1F));
  if ( v40 < 64 )
    v40 = 64;
  if ( (_DWORD)v29 != v40 )
  {
    v41 = (((4 * v40 / 3u + 1) | ((unsigned __int64)(4 * v40 / 3u + 1) >> 1)) >> 2)
        | (4 * v40 / 3u + 1)
        | ((unsigned __int64)(4 * v40 / 3u + 1) >> 1)
        | (((((4 * v40 / 3u + 1) | ((unsigned __int64)(4 * v40 / 3u + 1) >> 1)) >> 2)
          | (4 * v40 / 3u + 1)
          | ((unsigned __int64)(4 * v40 / 3u + 1) >> 1)) >> 4);
    v42 = (v41 >> 8) | v41;
    v43 = (v42 | (v42 >> 16)) + 1;
    v44 = 8 * ((v42 | (v42 >> 16)) + 1);
    goto LABEL_34;
  }
  *(_QWORD *)(a1 + 64) = 0;
  v50 = &v38[v29];
  do
  {
    if ( v38 )
      *v38 = -8;
    ++v38;
  }
  while ( v50 != v38 );
LABEL_14:
  v32 = *(_QWORD *)(a1 + 80);
  if ( v32 != *(_QWORD *)(a1 + 88) )
    *(_QWORD *)(a1 + 88) = v32;
  if ( (_DWORD)v54 )
  {
    v33 = v52;
    v34 = &v52[2 * (unsigned int)v54];
    do
    {
      if ( *v33 != -16 && *v33 != -8 )
      {
        v35 = v33[1];
        if ( v35 )
        {
          if ( (*(_BYTE *)(v35 + 8) & 1) == 0 )
            j___libc_free_0(*(_QWORD *)(v35 + 16));
          j_j___libc_free_0(v35, 552);
        }
      }
      v33 += 2;
    }
    while ( v34 != v33 );
  }
  j___libc_free_0(v52);
  return v26;
}
