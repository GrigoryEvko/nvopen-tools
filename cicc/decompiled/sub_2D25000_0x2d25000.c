// Function: sub_2D25000
// Address: 0x2d25000
//
__int64 __fastcall sub_2D25000(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v9; // rdx
  __int64 v10; // rcx
  unsigned __int64 v11; // r8
  __int64 v12; // r9
  int v13; // eax
  unsigned int v15; // edi
  __int64 v16; // r9
  unsigned int v17; // ecx
  unsigned int v18; // eax
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rsi
  _QWORD *v22; // rdx
  __int64 v23; // rcx
  int v25; // edi
  __int64 v26; // rax
  _QWORD *v27; // rcx
  __int64 v28; // rsi
  __int64 v29; // rcx
  int i; // eax
  int v31; // ecx
  __int64 v32; // r8
  __int64 v33; // r9
  __int64 v34; // rsi
  __int32 v35; // edx
  __int64 v36; // rsi
  __int64 v37; // rdi
  __int64 v38; // rcx
  __m128i v39; // xmm0
  __m128i *v40; // rcx
  __int64 v41; // rsi
  __int32 v42; // edx
  __int64 v43; // rsi
  unsigned int v44; // eax
  __m128i v45; // xmm1
  unsigned int v46; // esi
  int v47; // ecx
  unsigned __int64 v48; // r8
  int v51; // ecx
  unsigned __int64 v52; // rcx
  unsigned __int64 v53; // r8
  int v54; // eax
  __int64 v55; // r14
  int v56; // eax
  int v57; // [rsp+Ch] [rbp-D4h]
  int v58; // [rsp+Ch] [rbp-D4h]
  __int64 v59; // [rsp+10h] [rbp-D0h]
  __int64 v60; // [rsp+10h] [rbp-D0h]
  const void *v61; // [rsp+18h] [rbp-C8h]
  __m128i v62; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v63; // [rsp+30h] [rbp-B0h]
  __m128i v64; // [rsp+40h] [rbp-A0h] BYREF
  __int64 v65; // [rsp+50h] [rbp-90h]
  _BYTE *v66; // [rsp+60h] [rbp-80h] BYREF
  __int64 v67; // [rsp+68h] [rbp-78h]
  _BYTE v68[48]; // [rsp+70h] [rbp-70h] BYREF
  unsigned int v69; // [rsp+A0h] [rbp-40h]

  v61 = (const void *)(a1 + 16);
  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 72) = a1 + 88;
  *(_QWORD *)(a1 + 80) = 0x200000000LL;
  *(_QWORD *)(a1 + 144) = 0x200000000LL;
  *(_QWORD *)(a1 + 200) = a1 + 216;
  *(_QWORD *)(a1 + 208) = 0xC00000000LL;
  *(_QWORD *)(a1 + 8) = 0x600000000LL;
  *(_DWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 136) = a1 + 152;
  sub_2D24C50(a1, a4, a1 + 152, a4, a5, a6);
  v13 = *(_DWORD *)(a2 + 8);
  v66 = v68;
  v67 = 0x600000000LL;
  if ( v13 )
  {
    sub_2D23740((__int64)&v66, a2, v9, v10, v11, v12);
    v25 = v67;
    v9 = (unsigned int)v67;
    if ( *(_DWORD *)(a3 + 8) <= (unsigned int)v67 )
      v9 = *(unsigned int *)(a3 + 8);
    v69 = *(_DWORD *)(a2 + 64);
    if ( (_DWORD)v9 )
    {
      v26 = 0;
      v11 = 8LL * (unsigned int)v9;
      do
      {
        v27 = &v66[v26];
        v28 = *(_QWORD *)(*(_QWORD *)a3 + v26);
        v26 += 8;
        *v27 &= v28;
      }
      while ( v11 != v26 );
    }
    for ( ; (_DWORD)v9 != v25; *(_QWORD *)&v66[8 * v29] = 0 )
    {
      v29 = (unsigned int)v9;
      v9 = (unsigned int)(v9 + 1);
    }
  }
  else
  {
    v69 = *(_DWORD *)(a2 + 64);
  }
  _RCX = v69;
  if ( v69 )
  {
    v12 = (__int64)v66;
    v15 = (v69 - 1) >> 6;
    v11 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v69;
    v9 = 0;
    while ( 1 )
    {
      _RCX = *(_QWORD *)&v66[8 * v9];
      if ( v15 == (_DWORD)v9 )
        _RCX = v11 & *(_QWORD *)&v66[8 * v9];
      if ( _RCX )
        break;
      if ( v15 + 1 == ++v9 )
        goto LABEL_9;
    }
    __asm { tzcnt   rcx, rcx }
    for ( i = _RCX + ((_DWORD)v9 << 6); i != -1; i = _RCX + _RAX )
    {
      v31 = *(_DWORD *)(*(_QWORD *)(a2 + 200) + 4LL * i);
      if ( *(_DWORD *)(*(_QWORD *)(a3 + 200) + 4LL * i) != v31 )
        v31 = 2;
      v32 = 24LL * i;
      *(_DWORD *)(*(_QWORD *)(a1 + 200) + 4LL * i) = v31;
      v33 = v32 + *(_QWORD *)(a3 + 136);
      v34 = v32 + *(_QWORD *)(a2 + 136);
      v35 = *(_DWORD *)v34;
      if ( *(_DWORD *)v34 == *(_DWORD *)v33 && *(_QWORD *)(v34 + 8) == *(_QWORD *)(v33 + 8) )
      {
        if ( v35 == 1 )
        {
          v36 = 0;
          v37 = 0;
        }
        else
        {
          v58 = i;
          v60 = 24LL * i;
          sub_2D22B40((__int64)&v64, v34, v32 + *(_QWORD *)(a3 + 136));
          v35 = v64.m128i_i32[0];
          v37 = v64.m128i_i64[1];
          v36 = v65;
          i = v58;
          v32 = v60;
        }
      }
      else
      {
        v36 = 0;
        v37 = 0;
        v35 = 1;
      }
      v38 = *(_QWORD *)(a1 + 136);
      v64.m128i_i32[0] = v35;
      v64.m128i_i64[1] = v37;
      v39 = _mm_loadu_si128(&v64);
      v40 = (__m128i *)(v32 + v38);
      v65 = v36;
      v40[1].m128i_i64[0] = v36;
      *v40 = v39;
      v12 = v32 + *(_QWORD *)(a3 + 72);
      v41 = v32 + *(_QWORD *)(a2 + 72);
      v42 = *(_DWORD *)v41;
      if ( *(_DWORD *)v41 == *(_DWORD *)v12 && *(_QWORD *)(v41 + 8) == *(_QWORD *)(v12 + 8) )
      {
        if ( v42 == 1 )
        {
          _RCX = 0;
          v43 = 0;
        }
        else
        {
          v57 = i;
          v59 = v32;
          sub_2D22B40((__int64)&v62, v41, v32 + *(_QWORD *)(a3 + 72));
          v42 = v62.m128i_i32[0];
          v43 = v62.m128i_i64[1];
          _RCX = v63;
          i = v57;
          v32 = v59;
        }
      }
      else
      {
        _RCX = 0;
        v43 = 0;
        v42 = 1;
      }
      v62.m128i_i32[0] = v42;
      v11 = *(_QWORD *)(a1 + 72) + v32;
      v44 = i + 1;
      v62.m128i_i64[1] = v43;
      v45 = _mm_loadu_si128(&v62);
      *(_QWORD *)(v11 + 16) = _RCX;
      *(__m128i *)v11 = v45;
      v9 = v69;
      v63 = _RCX;
      if ( v69 == v44 )
        break;
      v12 = v44 >> 6;
      v46 = (v69 - 1) >> 6;
      if ( (unsigned int)v12 > v46 )
        break;
      v47 = 64 - (v44 & 0x3F);
      v48 = 0xFFFFFFFFFFFFFFFFLL >> v47;
      v9 = (unsigned int)v12;
      if ( v47 == 64 )
        v48 = 0;
      v11 = ~v48;
      while ( 1 )
      {
        _RAX = *(_QWORD *)&v66[8 * v9];
        _RCX = (unsigned int)v9;
        if ( (_DWORD)v12 == (_DWORD)v9 )
          _RAX = v11 & *(_QWORD *)&v66[8 * v9];
        if ( v46 == (_DWORD)v9 )
          _RAX &= 0xFFFFFFFFFFFFFFFFLL >> -(char)v69;
        if ( _RAX )
          break;
        if ( v46 < (unsigned int)++v9 )
          goto LABEL_9;
      }
      __asm { tzcnt   rax, rax }
      _RCX = (unsigned int)((_DWORD)v9 << 6);
    }
  }
LABEL_9:
  sub_2D23740(a1, a2, v9, _RCX, v11, v12);
  v17 = *(_DWORD *)(a2 + 64);
  *(_DWORD *)(a1 + 64) = v17;
  v18 = *(_DWORD *)(a3 + 64);
  if ( v17 < v18 )
  {
    v51 = v17 & 0x3F;
    if ( v51 )
      *(_QWORD *)(*(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8) - 8) &= ~(-1LL << v51);
    v52 = *(unsigned int *)(a1 + 8);
    *(_DWORD *)(a1 + 64) = v18;
    v53 = (v18 + 63) >> 6;
    if ( v53 != v52 )
    {
      if ( v53 >= v52 )
      {
        v55 = v53 - v52;
        if ( v53 > *(unsigned int *)(a1 + 12) )
        {
          sub_C8D5F0(a1, v61, v53, 8u, v53, v16);
          v52 = *(unsigned int *)(a1 + 8);
        }
        if ( 8 * v55 )
        {
          memset((void *)(*(_QWORD *)a1 + 8 * v52), 0, 8 * v55);
          LODWORD(v52) = *(_DWORD *)(a1 + 8);
        }
        v56 = *(_DWORD *)(a1 + 64);
        *(_DWORD *)(a1 + 8) = v55 + v52;
        v54 = v56 & 0x3F;
        if ( !v54 )
          goto LABEL_10;
LABEL_57:
        *(_QWORD *)(*(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8) - 8) &= ~(-1LL << v54);
        goto LABEL_10;
      }
      *(_DWORD *)(a1 + 8) = (v18 + 63) >> 6;
    }
    v54 = v18 & 0x3F;
    if ( !v54 )
      goto LABEL_10;
    goto LABEL_57;
  }
LABEL_10:
  v19 = 0;
  v20 = *(unsigned int *)(a3 + 8);
  v21 = 8 * v20;
  if ( (_DWORD)v20 )
  {
    do
    {
      v22 = (_QWORD *)(v19 + *(_QWORD *)a1);
      v23 = *(_QWORD *)(*(_QWORD *)a3 + v19);
      v19 += 8;
      *v22 |= v23;
    }
    while ( v19 != v21 );
  }
  if ( v66 != v68 )
    _libc_free((unsigned __int64)v66);
  return a1;
}
