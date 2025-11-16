// Function: sub_10E8150
// Address: 0x10e8150
//
__int64 __fastcall sub_10E8150(__int64 **a1, __int64 a2)
{
  unsigned int v2; // r13d
  __int64 v3; // rdx
  int v5; // r15d
  unsigned __int64 v6; // r15
  _BYTE *v8; // rax
  unsigned int v9; // r12d
  __int64 v10; // r13
  __int64 v12; // rdx
  int v13; // r14d
  unsigned __int64 v14; // r14
  int v17; // r15d

  if ( *(_BYTE *)a2 != 17 )
  {
    v3 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(a2 + 8) + 8LL) - 17;
    if ( (unsigned int)v3 > 1 || *(_BYTE *)a2 > 0x15u )
      return 0;
    goto LABEL_11;
  }
  v2 = *(_DWORD *)(a2 + 32);
  v3 = 1LL << ((unsigned __int8)v2 - 1);
  _RAX = *(_QWORD *)(a2 + 24);
  if ( v2 > 0x40 )
  {
    v3 &= *(_QWORD *)(_RAX + 8LL * ((v2 - 1) >> 6));
    if ( !v3 )
      goto LABEL_10;
    v17 = sub_C44500(a2 + 24);
    if ( v2 != v17 + (unsigned int)sub_C44590(a2 + 24) )
      goto LABEL_10;
LABEL_26:
    **a1 = a2 + 24;
    return 1;
  }
  if ( (v3 & _RAX) != 0 )
  {
    if ( v2 )
    {
      v5 = 64;
      v3 = ~(_RAX << (64 - (unsigned __int8)v2));
      if ( _RAX << (64 - (unsigned __int8)v2) != -1 )
      {
        _BitScanReverse64(&v6, v3);
        v5 = v6 ^ 0x3F;
      }
    }
    else
    {
      v5 = 0;
    }
    __asm { tzcnt   rax, rax }
    if ( (unsigned int)_RAX > v2 )
      LODWORD(_RAX) = *(_DWORD *)(a2 + 32);
    if ( v2 == v5 + (_DWORD)_RAX )
      goto LABEL_26;
  }
LABEL_10:
  if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(a2 + 8) + 8LL) - 17 > 1 )
    return 0;
LABEL_11:
  v8 = sub_AD7630(a2, 1, v3);
  if ( !v8 || *v8 != 17 )
    return 0;
  v9 = *((_DWORD *)v8 + 8);
  v10 = (__int64)(v8 + 24);
  _RAX = *((_QWORD *)v8 + 3);
  v12 = 1LL << ((unsigned __int8)v9 - 1);
  if ( v9 > 0x40 )
  {
    if ( (*(_QWORD *)(_RAX + 8LL * ((v9 - 1) >> 6)) & v12) != 0 )
    {
      v13 = sub_C44500(v10);
      LODWORD(_RAX) = sub_C44590(v10);
      goto LABEL_29;
    }
    return 0;
  }
  if ( (v12 & _RAX) == 0 )
    return 0;
  if ( v9 )
  {
    v13 = 64;
    if ( _RAX << (64 - (unsigned __int8)v9) != -1 )
    {
      _BitScanReverse64(&v14, ~(_RAX << (64 - (unsigned __int8)v9)));
      v13 = v14 ^ 0x3F;
    }
  }
  else
  {
    v13 = 0;
  }
  __asm { tzcnt   rax, rax }
  if ( (unsigned int)_RAX > v9 )
    LODWORD(_RAX) = v9;
LABEL_29:
  if ( v9 != v13 + (_DWORD)_RAX )
    return 0;
  **a1 = v10;
  return 1;
}
