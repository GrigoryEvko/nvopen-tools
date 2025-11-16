// Function: sub_10A8010
// Address: 0x10a8010
//
__int64 __fastcall sub_10A8010(_QWORD **a1, int a2, unsigned __int8 *a3)
{
  __int64 v4; // rcx
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rcx
  __int64 v8; // rcx
  __int64 v9; // r12
  unsigned int v10; // r13d
  __int64 v11; // rsi
  int v13; // r15d
  unsigned __int64 v14; // rsi
  int v15; // esi
  __int64 v17; // rax
  __int64 v18; // r12
  unsigned int v19; // r13d
  __int64 v20; // r14
  __int64 v21; // rsi
  int v23; // r15d
  unsigned __int64 v24; // r15
  _BYTE *v26; // rax
  unsigned int v27; // r13d
  __int64 v29; // rsi
  int v30; // r12d
  unsigned __int64 v31; // rsi
  int v32; // esi
  _BYTE *v34; // rax
  _BYTE *v35; // r12
  char v36; // al
  unsigned __int8 *v37; // [rsp-40h] [rbp-40h]
  unsigned __int8 *v38; // [rsp-40h] [rbp-40h]
  unsigned __int8 *v39; // [rsp-40h] [rbp-40h]
  unsigned __int8 *v40; // [rsp-40h] [rbp-40h]

  if ( a2 + 29 != *a3 )
    return 0;
  v4 = *((_QWORD *)a3 - 8);
  v5 = *(_QWORD *)(v4 + 16);
  if ( !v5 )
    goto LABEL_4;
  if ( *(_QWORD *)(v5 + 8) )
    goto LABEL_4;
  if ( *(_BYTE *)v4 != 46 )
    goto LABEL_4;
  v17 = *(_QWORD *)(v4 - 64);
  if ( !v17 )
    goto LABEL_4;
  **a1 = v17;
  v18 = *(_QWORD *)(v4 - 32);
  if ( *(_BYTE *)v18 != 17 )
  {
    if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v18 + 8) + 8LL) - 17 > 1 || *(_BYTE *)v18 > 0x15u )
      goto LABEL_4;
    goto LABEL_38;
  }
  v19 = *(_DWORD *)(v18 + 32);
  v20 = v18 + 24;
  v21 = 1LL << ((unsigned __int8)v19 - 1);
  _RAX = *(_QWORD *)(v18 + 24);
  if ( v19 > 0x40 )
  {
    if ( (*(_QWORD *)(_RAX + 8LL * ((v19 - 1) >> 6)) & v21) == 0 )
      goto LABEL_49;
    v39 = a3;
    v23 = sub_C44500(v18 + 24);
    LODWORD(_RAX) = sub_C44590(v18 + 24);
    a3 = v39;
  }
  else
  {
    if ( (v21 & _RAX) == 0 )
      goto LABEL_49;
    if ( v19 )
    {
      v23 = 64;
      if ( _RAX << (64 - (unsigned __int8)v19) != -1 )
      {
        _BitScanReverse64(&v24, ~(_RAX << (64 - (unsigned __int8)v19)));
        v23 = v24 ^ 0x3F;
      }
    }
    else
    {
      v23 = 0;
    }
    __asm { tzcnt   rax, rax }
    if ( (unsigned int)_RAX > v19 )
      LODWORD(_RAX) = *(_DWORD *)(v18 + 32);
  }
  if ( v19 == v23 + (_DWORD)_RAX )
    goto LABEL_34;
LABEL_49:
  if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v18 + 8) + 8LL) - 17 > 1 )
    goto LABEL_4;
LABEL_38:
  v37 = a3;
  v26 = sub_AD7630(v18, 1, (__int64)a3);
  a3 = v37;
  if ( !v26 || *v26 != 17 )
    goto LABEL_4;
  v27 = *((_DWORD *)v26 + 8);
  v20 = (__int64)(v26 + 24);
  _RAX = *((_QWORD *)v26 + 3);
  v29 = 1LL << ((unsigned __int8)v27 - 1);
  if ( v27 > 0x40 )
  {
    if ( (*(_QWORD *)(_RAX + 8LL * ((v27 - 1) >> 6)) & v29) != 0 )
    {
      v30 = sub_C44500(v20);
      LODWORD(_RAX) = sub_C44590(v20);
      a3 = v37;
      goto LABEL_47;
    }
LABEL_4:
    v6 = *((_QWORD *)a3 - 4);
    goto LABEL_5;
  }
  if ( (v29 & _RAX) == 0 )
    goto LABEL_4;
  if ( v27 )
  {
    v30 = 64;
    _BitScanReverse64(&v31, ~(_RAX << (64 - (unsigned __int8)v27)));
    v32 = v31 ^ 0x3F;
    if ( _RAX << (64 - (unsigned __int8)v27) != -1 )
      v30 = v32;
  }
  else
  {
    v30 = 0;
  }
  __asm { tzcnt   rax, rax }
  if ( (unsigned int)_RAX > v27 )
    LODWORD(_RAX) = v27;
LABEL_47:
  if ( v27 != v30 + (_DWORD)_RAX )
    goto LABEL_4;
LABEL_34:
  *a1[1] = v20;
  v6 = *((_QWORD *)a3 - 4);
  if ( v6 )
    goto LABEL_35;
LABEL_5:
  v7 = *(_QWORD *)(v6 + 16);
  if ( !v7 )
    return 0;
  if ( *(_QWORD *)(v7 + 8) )
    return 0;
  if ( *(_BYTE *)v6 != 46 )
    return 0;
  v8 = *(_QWORD *)(v6 - 64);
  if ( !v8 )
    return 0;
  **a1 = v8;
  v9 = *(_QWORD *)(v6 - 32);
  if ( *(_BYTE *)v9 == 17 )
  {
    v10 = *(_DWORD *)(v9 + 32);
    v11 = 1LL << ((unsigned __int8)v10 - 1);
    _RAX = *(_QWORD *)(v9 + 24);
    if ( v10 > 0x40 )
    {
      if ( (*(_QWORD *)(_RAX + 8LL * ((v10 - 1) >> 6)) & v11) != 0 )
      {
        v40 = a3;
        v13 = sub_C44500(v9 + 24);
        LODWORD(_RAX) = sub_C44590(v9 + 24);
        a3 = v40;
        goto LABEL_18;
      }
    }
    else if ( (v11 & _RAX) != 0 )
    {
      if ( v10 )
      {
        v13 = 64;
        _BitScanReverse64(&v14, ~(_RAX << (64 - (unsigned __int8)v10)));
        v15 = v14 ^ 0x3F;
        if ( _RAX << (64 - (unsigned __int8)v10) != -1 )
          v13 = v15;
      }
      else
      {
        v13 = 0;
      }
      __asm { tzcnt   rax, rax }
      if ( (unsigned int)_RAX > v10 )
        LODWORD(_RAX) = *(_DWORD *)(v9 + 32);
LABEL_18:
      if ( v10 == v13 + (_DWORD)_RAX )
      {
        *a1[1] = v9 + 24;
        goto LABEL_20;
      }
    }
    if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v9 + 8) + 8LL) - 17 > 1 )
      return 0;
    goto LABEL_53;
  }
  if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v9 + 8) + 8LL) - 17 > 1 || *(_BYTE *)v9 > 0x15u )
    return 0;
LABEL_53:
  v38 = a3;
  v34 = sub_AD7630(v9, 1, (__int64)a3);
  if ( !v34 )
    return 0;
  if ( *v34 != 17 )
    return 0;
  v35 = v34 + 24;
  v36 = sub_109DE70((__int64)(v34 + 24));
  a3 = v38;
  if ( !v36 )
    return 0;
  *a1[1] = v35;
LABEL_20:
  v6 = *((_QWORD *)a3 - 8);
  if ( !v6 )
    return 0;
LABEL_35:
  *a1[2] = v6;
  return 1;
}
