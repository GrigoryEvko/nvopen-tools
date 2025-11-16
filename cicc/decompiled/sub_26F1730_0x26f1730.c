// Function: sub_26F1730
// Address: 0x26f1730
//
bool __fastcall sub_26F1730(_QWORD *a1, _BYTE *a2)
{
  __int64 v4; // rax
  __int64 v5; // rsi
  int v6; // edx
  int v7; // ecx
  unsigned int v8; // edx
  __int64 v9; // rdi
  bool result; // al
  int v11; // r8d
  __int64 v12; // rax
  __int64 v13; // rsi
  int v14; // eax
  int v15; // ecx
  unsigned int v16; // edx
  _BYTE *v17; // rdi
  int v18; // eax
  _BYTE *v19; // r12
  __int64 v20; // rax
  unsigned __int8 v21; // dl
  unsigned __int8 **v22; // rax
  unsigned __int8 *v23; // rdx
  _BYTE *v24; // rdi

  v4 = sub_B326A0((__int64)a2);
  if ( v4 )
  {
    v5 = *(_QWORD *)(*a1 + 8LL);
    v6 = *(_DWORD *)(*a1 + 24LL);
    if ( v6 )
    {
      v7 = v6 - 1;
      v8 = (v6 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v9 = *(_QWORD *)(v5 + 8LL * v8);
      if ( v4 == v9 )
        return 1;
      v11 = 1;
      while ( v9 != -4096 )
      {
        v8 = v7 & (v11 + v8);
        v9 = *(_QWORD *)(v5 + 8LL * v8);
        if ( v4 == v9 )
          return 1;
        ++v11;
      }
    }
  }
  if ( *a2 )
  {
    v19 = (_BYTE *)sub_B32590((__int64)a2);
    result = 0;
    if ( !v19 || *v19 != 3 )
      return result;
    if ( (v19[7] & 0x20) != 0 )
    {
      v20 = sub_B91C10((__int64)v19, 22);
      if ( v20 )
      {
        v21 = *(_BYTE *)(v20 - 16);
        v22 = (v21 & 2) != 0
            ? *(unsigned __int8 ***)(v20 - 32)
            : (unsigned __int8 **)(v20 - 8LL * ((v21 >> 2) & 0xF) - 16);
        v23 = *v22;
        if ( *v22 )
        {
          if ( (unsigned int)*v23 - 1 <= 1 )
          {
            v24 = (_BYTE *)*((_QWORD *)v23 + 17);
            if ( ((unsigned __int8)(*v24 - 2) <= 1u || !*v24) && (v24[7] & 0x20) != 0 && sub_B91C10((__int64)v24, 19) )
              return 1;
          }
        }
      }
      if ( (v19[7] & 0x20) != 0 )
        return sub_B91C10((__int64)v19, 19) != 0;
    }
    return 0;
  }
  v12 = a1[1];
  v13 = *(_QWORD *)(v12 + 8);
  v14 = *(_DWORD *)(v12 + 24);
  if ( !v14 )
    return 0;
  v15 = v14 - 1;
  v16 = (v14 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v17 = *(_BYTE **)(v13 + 8LL * v16);
  result = 1;
  if ( a2 != v17 )
  {
    v18 = 1;
    while ( v17 != (_BYTE *)-4096LL )
    {
      v16 = v15 & (v18 + v16);
      v17 = *(_BYTE **)(v13 + 8LL * v16);
      if ( a2 == v17 )
        return 1;
      ++v18;
    }
    return 0;
  }
  return result;
}
