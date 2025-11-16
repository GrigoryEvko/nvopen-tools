// Function: sub_1F417D0
// Address: 0x1f417d0
//
__int64 __fastcall sub_1F417D0(__int64 a1, __int64 a2, unsigned __int8 a3)
{
  __int64 v3; // r14
  __int64 v4; // rbx
  unsigned int v5; // r15d
  __int64 v6; // rax
  size_t v7; // rdx
  size_t v8; // r8
  char *v9; // r12
  __int64 v10; // rcx
  _WORD *v11; // rdx
  _WORD *v12; // rax
  __int64 v13; // rdi
  __int64 v14; // r13
  char *v15; // r15
  unsigned int v16; // ebx
  __int64 v17; // r10
  _WORD *v18; // r9
  __int64 v19; // rax
  unsigned int v20; // ecx
  __int64 v21; // rdx
  __int64 v22; // rax
  char *v23; // r11
  __int64 v24; // rax
  int v25; // ecx
  __int64 v26; // rsi
  __int64 v27; // r8
  unsigned __int64 v28; // rdx
  size_t v29; // r8
  unsigned int v30; // r10d
  unsigned int v34; // r15d
  __int64 v35; // rax
  unsigned int v36; // ebx
  __int64 v37; // r13
  __int64 v38; // rdx
  int v39; // eax
  unsigned int v40; // eax
  unsigned int v41; // r8d
  __int64 v42; // rdx
  int v43; // ecx
  unsigned __int64 v44; // r10
  unsigned __int64 v45; // r10
  __int64 v48; // rax
  __int64 v50; // [rsp+8h] [rbp-78h]
  size_t v51; // [rsp+10h] [rbp-70h]
  __int64 v52; // [rsp+20h] [rbp-60h]
  size_t v54; // [rsp+38h] [rbp-48h]
  __int64 v55; // [rsp+38h] [rbp-48h]
  int v56; // [rsp+38h] [rbp-48h]
  size_t n; // [rsp+40h] [rbp-40h]
  unsigned int na; // [rsp+40h] [rbp-40h]
  size_t nb; // [rsp+40h] [rbp-40h]
  size_t nc; // [rsp+40h] [rbp-40h]

  v3 = *(_QWORD *)(a1 + 8LL * a3 + 120);
  if ( v3 )
  {
    v4 = *(_QWORD *)(a2 + 264);
    v54 = *(_QWORD *)(a2 + 256);
    v52 = (__int64)(v4 - v54) >> 3;
    v5 = (unsigned int)(v52 + 63) >> 6;
    n = 8LL * v5;
    v6 = malloc(n);
    v7 = n;
    v8 = v54;
    v9 = (char *)v6;
    if ( v6 )
    {
      LODWORD(v10) = v52;
    }
    else if ( n || (v48 = malloc(1u), v7 = 0, !v48) )
    {
      nc = v7;
      sub_16BD1C0("Allocation failed", 1u);
      v4 = *(_QWORD *)(a2 + 264);
      v8 = *(_QWORD *)(a2 + 256);
      v7 = nc;
      v10 = (__int64)(v4 - v8) >> 3;
    }
    else
    {
      LODWORD(v10) = v52;
      v8 = v54;
      v9 = (char *)v48;
    }
    if ( v5 )
    {
      v56 = v10;
      nb = v8;
      memset(v9, 0, v7);
      LODWORD(v10) = v56;
      v8 = nb;
    }
    v11 = *(_WORD **)(v3 + 16);
    v51 = v8;
    v12 = 0;
    v50 = v4;
    v13 = 4LL * ((unsigned int)(v10 + 31) >> 5) + *(_QWORD *)(v3 + 8);
    v14 = 4LL * ((unsigned int)(v10 + 31) >> 5);
    if ( *v11 )
      v12 = v11 + 1;
    v15 = &v9[8 * v5 - 8];
    v16 = (unsigned int)(v52 + 31) >> 5;
    na = ((v16 - 2) >> 1) + 1;
    v17 = 8LL * na;
    while ( 1 )
    {
      v18 = v12;
      if ( !v12 )
        break;
      while ( 1 )
      {
        v19 = 0;
        if ( v16 <= 1 )
        {
          v20 = (unsigned int)(v52 + 31) >> 5;
          v22 = v13;
          v21 = 0;
        }
        else
        {
          do
          {
            *(_QWORD *)&v9[v19] |= *(_QWORD *)(v13 + v19);
            v19 += 8;
          }
          while ( v17 != v19 );
          v20 = (((_DWORD)v52 + 31) & 0x20) != 0;
          v21 = na;
          v22 = v13 + v17;
        }
        if ( v20 )
        {
          v23 = &v9[8 * v21];
          v24 = v22 + 4;
          v25 = 0;
          v26 = *(_QWORD *)v23;
          v27 = v24;
          while ( 1 )
          {
            v28 = (unsigned __int64)*(unsigned int *)(v24 - 4) << v25;
            v25 += 32;
            v26 |= v28;
            if ( v24 == v27 )
              break;
            v24 += 4;
          }
          *(_QWORD *)v23 = v26;
        }
        if ( (v52 & 0x3F) != 0 )
          *(_QWORD *)v15 &= ~(-1LL << v52);
        ++v18;
        v13 += v14;
        v12 = 0;
        if ( !*(v18 - 1) )
          break;
        if ( !v18 )
          goto LABEL_22;
      }
    }
LABEL_22:
    v29 = v51;
    if ( !(_DWORD)v52 )
      goto LABEL_28;
    v30 = (unsigned int)(v52 - 1) >> 6;
    while ( 1 )
    {
      _RDX = *(_QWORD *)&v9[8 * (_QWORD)v18];
      if ( v30 == (_DWORD)v18 )
        _RDX = (0xFFFFFFFFFFFFFFFFLL >> -(char)v52) & *(_QWORD *)&v9[8 * (_QWORD)v18];
      if ( _RDX )
        break;
      v18 = (_WORD *)((char *)v18 + 1);
      if ( (_WORD *)(v30 + 1) == v18 )
        goto LABEL_28;
    }
    __asm { tzcnt   rdx, rdx }
    v34 = ((_DWORD)v18 << 6) + _RDX;
    if ( v34 == -1 )
    {
LABEL_28:
      _libc_free((unsigned __int64)v9);
    }
    else
    {
      v35 = v50;
      v36 = (unsigned int)(v52 - 1) >> 6;
      v37 = v3;
      while ( 1 )
      {
        v38 = *(_QWORD *)(v29 + 8LL * v34);
        v39 = *(_DWORD *)(a2 + 288) * ((__int64)(v35 - v29) >> 3);
        if ( *(_DWORD *)(*(_QWORD *)(a2 + 280)
                       + 24LL * (v39 + (unsigned int)*(unsigned __int16 *)(*(_QWORD *)v37 + 24LL))
                       + 4) >> 3 < *(_DWORD *)(*(_QWORD *)(a2 + 280)
                                             + 24LL * ((unsigned int)*(unsigned __int16 *)(*(_QWORD *)v38 + 24LL) + v39)
                                             + 4) >> 3 )
        {
          v55 = *(_QWORD *)(v29 + 8LL * v34);
          if ( (unsigned __int8)sub_1F41770(a1, a2, v38) )
            v37 = v55;
        }
        v40 = v34 + 1;
        if ( (_DWORD)v52 == v34 + 1 )
          break;
        v41 = v40 >> 6;
        if ( v36 < v40 >> 6 )
          break;
        v42 = v41;
        v43 = 64 - (v40 & 0x3F);
        v44 = 0xFFFFFFFFFFFFFFFFLL >> v43;
        if ( v43 == 64 )
          v44 = 0;
        v45 = ~v44;
        while ( 1 )
        {
          _RAX = *(_QWORD *)&v9[8 * v42];
          if ( v41 == (_DWORD)v42 )
            _RAX = v45 & *(_QWORD *)&v9[8 * v42];
          if ( v36 == (_DWORD)v42 )
            _RAX &= 0xFFFFFFFFFFFFFFFFLL >> -(char)v52;
          if ( _RAX )
            break;
          if ( v36 < (unsigned int)++v42 )
            goto LABEL_48;
        }
        __asm { tzcnt   rax, rax }
        v34 = ((_DWORD)v42 << 6) + _RAX;
        if ( v34 == -1 )
          break;
        v29 = *(_QWORD *)(a2 + 256);
        v35 = *(_QWORD *)(a2 + 264);
      }
LABEL_48:
      v3 = v37;
      _libc_free((unsigned __int64)v9);
    }
  }
  return v3;
}
