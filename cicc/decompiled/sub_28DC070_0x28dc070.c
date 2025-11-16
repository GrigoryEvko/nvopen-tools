// Function: sub_28DC070
// Address: 0x28dc070
//
void __fastcall sub_28DC070(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r13
  unsigned int v5; // esi
  int v6; // eax
  __int64 *v7; // rdx
  int v8; // eax
  _QWORD *v9; // r14
  char v10; // dl
  char v11; // r13
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rsi
  int v15; // eax
  int v16; // edi
  unsigned int v17; // edx
  __int64 *v18; // rax
  __int64 v19; // r8
  unsigned __int8 *v20; // rax
  __int64 v21; // rsi
  int v22; // eax
  __int64 v23; // rdi
  int v24; // ecx
  unsigned int v25; // edx
  __int64 *v26; // rax
  __int64 v27; // r8
  unsigned int v28; // ecx
  __int64 v29; // rdx
  __int64 v30; // rax
  __int64 *v31; // rax
  __int64 v32; // r8
  __int64 *v33; // r10
  int v35; // ecx
  int v37; // ecx
  unsigned int v38; // esi
  char v39; // al
  unsigned __int64 v40; // rdi
  unsigned int v41; // ecx
  unsigned __int64 v42; // rax
  char v43; // dl
  unsigned int v44; // edx
  unsigned int v47; // ecx
  int v50; // ecx
  int v52; // ecx
  unsigned int v54; // esi
  __int64 v55; // rdi
  unsigned int v56; // esi
  unsigned int v57; // edx
  __int64 *v58; // rax
  __int64 v59; // r8
  unsigned int v60; // edx
  int v61; // eax
  int v62; // r9d
  int v63; // eax
  int v64; // eax
  int v65; // r9d
  int v66; // r9d
  __int64 v67; // [rsp+8h] [rbp-68h] BYREF
  __int64 *v68; // [rsp+10h] [rbp-60h] BYREF
  __int64 *v69; // [rsp+18h] [rbp-58h] BYREF
  __m128i v70; // [rsp+20h] [rbp-50h] BYREF

  v3 = a1 + 2152;
  v67 = a3;
  v70.m128i_i64[0] = a2;
  v70.m128i_i64[1] = a3;
  if ( (unsigned __int8)sub_28CE830(a1 + 2152, v70.m128i_i64, &v68) )
    return;
  v5 = *(_DWORD *)(a1 + 2176);
  v6 = *(_DWORD *)(a1 + 2168);
  v7 = v68;
  ++*(_QWORD *)(a1 + 2152);
  v8 = v6 + 1;
  v69 = v7;
  if ( 4 * v8 >= 3 * v5 )
  {
    v5 *= 2;
    goto LABEL_48;
  }
  if ( v5 - *(_DWORD *)(a1 + 2172) - v8 <= v5 >> 3 )
  {
LABEL_48:
    sub_2805990(v3, v5);
    sub_28CE830(v3, v70.m128i_i64, &v69);
    v7 = v69;
    v8 = *(_DWORD *)(a1 + 2168) + 1;
  }
  *(_DWORD *)(a1 + 2168) = v8;
  if ( *v7 != -4096 || v7[1] != -4096 )
    --*(_DWORD *)(a1 + 2172);
  *(__m128i *)v7 = _mm_loadu_si128(&v70);
  v9 = sub_AE6EC0(a1 + 2184, v67);
  v11 = v10;
  v12 = sub_254BB00(a1 + 2184);
  v70.m128i_i64[0] = (__int64)v9;
  v70.m128i_i64[1] = v12;
  sub_254BBF0((__int64)&v70);
  if ( v11 )
  {
    v54 = *(_DWORD *)(a1 + 2376);
    v55 = *(_QWORD *)(a1 + 2360);
    if ( v54 )
    {
      v56 = v54 - 1;
      v57 = v56 & (((unsigned int)v67 >> 9) ^ ((unsigned int)v67 >> 4));
      v58 = (__int64 *)(v55 + 16LL * v57);
      v59 = *v58;
      if ( v67 == *v58 )
      {
LABEL_40:
        v54 = *((_DWORD *)v58 + 2);
        v60 = *((_DWORD *)v58 + 3);
      }
      else
      {
        v63 = 1;
        while ( v59 != -4096 )
        {
          v66 = v63 + 1;
          v57 = v56 & (v63 + v57);
          v58 = (__int64 *)(v55 + 16LL * v57);
          v59 = *v58;
          if ( v67 == *v58 )
            goto LABEL_40;
          v63 = v66;
        }
        v60 = 0;
        v54 = 0;
      }
    }
    else
    {
      v60 = 0;
    }
    sub_28C7AB0((_QWORD *)(a1 + 2280), v54, v60);
    return;
  }
  v13 = *(_QWORD *)(a1 + 32);
  v14 = *(_QWORD *)(v13 + 40);
  v15 = *(_DWORD *)(v13 + 56);
  if ( v15 )
  {
    v16 = v15 - 1;
    v17 = (v15 - 1) & (((unsigned int)v67 >> 9) ^ ((unsigned int)v67 >> 4));
    v18 = (__int64 *)(v14 + 16LL * v17);
    v19 = *v18;
    if ( v67 == *v18 )
    {
LABEL_10:
      v20 = (unsigned __int8 *)v18[1];
      if ( !v20 )
        goto LABEL_16;
      if ( (unsigned int)*v20 - 26 > 1 )
      {
        v70.m128i_i64[0] = (__int64)v20;
        v26 = sub_28CBE90(a1 + 2416, v70.m128i_i64);
        if ( v26 )
          goto LABEL_14;
      }
      else
      {
        v21 = *((_QWORD *)v20 + 9);
        v22 = *(_DWORD *)(a1 + 2440);
        v23 = *(_QWORD *)(a1 + 2424);
        if ( v22 )
        {
          v24 = v22 - 1;
          v25 = (v22 - 1) & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
          v26 = (__int64 *)(v23 + 16LL * v25);
          v27 = *v26;
          if ( v21 == *v26 )
          {
LABEL_14:
            v28 = *((_DWORD *)v26 + 2);
            v29 = 1LL << v28;
            v30 = 8LL * (v28 >> 6);
LABEL_15:
            *(_QWORD *)(*(_QWORD *)(a1 + 2280) + v30) |= v29;
            goto LABEL_16;
          }
          v64 = 1;
          while ( v27 != -4096 )
          {
            v65 = v64 + 1;
            v25 = v24 & (v64 + v25);
            v26 = (__int64 *)(v23 + 16LL * v25);
            v27 = *v26;
            if ( v21 == *v26 )
              goto LABEL_14;
            v64 = v65;
          }
        }
      }
      v29 = 1;
      v30 = 0;
      goto LABEL_15;
    }
    v61 = 1;
    while ( v19 != -4096 )
    {
      v62 = v61 + 1;
      v17 = v16 & (v61 + v17);
      v18 = (__int64 *)(v14 + 16LL * v17);
      v19 = *v18;
      if ( v67 == *v18 )
        goto LABEL_10;
      v61 = v62;
    }
  }
LABEL_16:
  v31 = sub_28D5740(a1 + 1824, &v67);
  v32 = *v31;
  v33 = v31;
  if ( (__int64 *)*v31 == v31 )
  {
    v40 = 0;
    v38 = 0;
    v39 = 1;
  }
  else
  {
    _RAX = *(_QWORD *)(v32 + 24);
    if ( _RAX )
    {
      v35 = 0;
    }
    else
    {
      _RAX = *(_QWORD *)(v32 + 32);
      if ( !_RAX )
LABEL_64:
        BUG();
      v35 = 64;
    }
    __asm { tzcnt   rax, rax }
    v37 = _RAX + v35;
    v38 = v37 + (*(_DWORD *)(v32 + 16) << 7);
    v39 = 0;
    v40 = *(_QWORD *)(v32 + 8LL * ((v38 >> 6) & 1) + 24) >> v37;
  }
  if ( !v39 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v41 = v38 + 1;
        *(_QWORD *)(*(_QWORD *)(a1 + 2280) + 8LL * (v38 >> 6)) |= 1LL << v38;
        v42 = v40 >> 1;
        if ( !(v40 >> 1) )
          break;
        while ( 1 )
        {
          v43 = v42;
          v40 = v42;
          v38 = v41;
          v42 >>= 1;
          ++v41;
          if ( (v43 & 1) != 0 )
            break;
          if ( !v42 )
            goto LABEL_25;
        }
      }
LABEL_25:
      v44 = v41 & 0x7F;
      if ( *(_QWORD *)(v32 + 8LL * (v44 >> 6) + 24) & (-1LL << v41) )
        break;
      if ( (unsigned __int8)(v41 & 0x7F) >> 6 != 1 && (_RAX = *(_QWORD *)(v32 + 32)) != 0 )
      {
        __asm { tzcnt   rax, rax }
        v47 = _RAX + 64;
LABEL_27:
        if ( !v44 )
          goto LABEL_32;
        v38 = v47 + (*(_DWORD *)(v32 + 16) << 7);
        v40 = *(_QWORD *)(v32 + 8LL * (v47 >> 6) + 24) >> v47;
      }
      else
      {
LABEL_32:
        v32 = *(_QWORD *)v32;
        if ( v33 == (__int64 *)v32 )
          return;
        _RAX = *(_QWORD *)(v32 + 24);
        if ( _RAX )
        {
          v50 = 0;
        }
        else
        {
          _RAX = *(_QWORD *)(v32 + 32);
          if ( !_RAX )
            goto LABEL_64;
          v50 = 64;
        }
        __asm { tzcnt   rax, rax }
        v52 = _RAX + v50;
        v38 = v52 + (*(_DWORD *)(v32 + 16) << 7);
        v40 = *(_QWORD *)(v32 + 8LL * ((v38 >> 6) & 1) + 24) >> v52;
      }
    }
    __asm { tzcnt   rax, rax }
    v47 = _RAX + (v41 & 0x40);
    goto LABEL_27;
  }
}
