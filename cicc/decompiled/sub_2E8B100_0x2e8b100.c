// Function: sub_2E8B100
// Address: 0x2e8b100
//
__int64 __fastcall sub_2E8B100(__int64 a1, __int64 a2, __int64 a3, __int64 a4, _QWORD *a5)
{
  int v5; // eax
  __int64 v6; // rax
  int v7; // eax
  __int64 v8; // rax
  __int64 v9; // rax
  unsigned __int64 v10; // rdx
  int v11; // eax
  _QWORD *v12; // rax
  __int64 v13; // rcx
  _QWORD *v14; // rdx
  __int64 v15; // rdi
  __int64 v16; // rcx
  _QWORD *v17; // rdi
  int v19; // eax
  char v20; // al
  int v21; // eax
  __int64 v22; // rsi
  __int64 v23; // rsi
  __int64 v24; // rsi

  if ( (unsigned int)*(unsigned __int16 *)(a1 + 68) - 1 > 1 || (*(_BYTE *)(*(_QWORD *)(a1 + 32) + 64LL) & 0x10) == 0 )
  {
    v5 = *(_DWORD *)(a1 + 44);
    if ( (v5 & 4) != 0 || (v5 & 8) == 0 )
      v6 = (*(_QWORD *)(*(_QWORD *)(a1 + 16) + 24LL) >> 20) & 1LL;
    else
      LOBYTE(v6) = sub_2E88A90(a1, 0x100000, 1);
    if ( !(_BYTE)v6
      && ((unsigned int)*(unsigned __int16 *)(a1 + 68) - 1 > 1 || (*(_BYTE *)(*(_QWORD *)(a1 + 32) + 64LL) & 8) == 0) )
    {
      v7 = *(_DWORD *)(a1 + 44);
      if ( (v7 & 4) == 0 && (v7 & 8) != 0 )
        LOBYTE(v8) = sub_2E88A90(a1, 0x80000, 1);
      else
        v8 = (*(_QWORD *)(*(_QWORD *)(a1 + 16) + 24LL) >> 19) & 1LL;
      if ( !(_BYTE)v8 )
      {
        v19 = *(_DWORD *)(a1 + 44);
        if ( (v19 & 4) == 0 && (v19 & 8) != 0 )
          v20 = sub_2E88A90(a1, 128, 1);
        else
          v20 = (unsigned __int8)*(_QWORD *)(*(_QWORD *)(a1 + 16) + 24LL) >> 7;
        if ( !v20 )
        {
          LOBYTE(v21) = sub_2E8B090(a1);
          LODWORD(a5) = v21;
          if ( !(_BYTE)v21 )
            return (unsigned int)a5;
        }
      }
    }
  }
  v9 = *(_QWORD *)(a1 + 48);
  v10 = v9 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (v9 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    goto LABEL_22;
  if ( (v9 & 7) != 0 )
  {
    if ( (v9 & 7) == 3 && *(_DWORD *)v10 )
    {
      v11 = v9 & 7;
      if ( v11 )
        goto LABEL_15;
LABEL_28:
      *(_QWORD *)(a1 + 48) = v10;
      v12 = (_QWORD *)(a1 + 48);
      v14 = (_QWORD *)(a1 + 56);
      goto LABEL_29;
    }
LABEL_22:
    LODWORD(a5) = 1;
    return (unsigned int)a5;
  }
  *(_QWORD *)(a1 + 48) = v10;
  v11 = 0;
  if ( (v10 & 7) == 0 )
    goto LABEL_28;
LABEL_15:
  LODWORD(a5) = 0;
  if ( v11 != 3 )
    return (unsigned int)a5;
  v12 = (_QWORD *)(v10 + 16);
  v13 = 8LL * *(int *)v10;
  v14 = (_QWORD *)(v10 + 16 + v13);
  v15 = v13 >> 5;
  v16 = v13 >> 3;
  if ( v15 > 0 )
  {
    v17 = &v12[4 * v15];
    while ( (*(_BYTE *)(*v12 + 37LL) & 0xFu) <= 1 && (*(_BYTE *)(*v12 + 32LL) & 4) == 0 )
    {
      v22 = v12[1];
      a5 = v12 + 1;
      if ( (*(_BYTE *)(v22 + 37) & 0xFu) > 1 || (*(_BYTE *)(v22 + 32) & 4) != 0 )
      {
        LOBYTE(a5) = a5 != v14;
        return (unsigned int)a5;
      }
      v23 = v12[2];
      a5 = v12 + 2;
      if ( (*(_BYTE *)(v23 + 37) & 0xFu) > 1
        || (*(_BYTE *)(v23 + 32) & 4) != 0
        || (v24 = v12[3], a5 = v12 + 3, (*(_BYTE *)(v24 + 37) & 0xFu) > 1)
        || (*(_BYTE *)(v24 + 32) & 4) != 0 )
      {
        LOBYTE(a5) = v14 != a5;
        return (unsigned int)a5;
      }
      v12 += 4;
      if ( v12 == v17 )
      {
        v16 = v14 - v12;
        goto LABEL_48;
      }
    }
    goto LABEL_20;
  }
LABEL_48:
  if ( v16 != 2 )
  {
    if ( v16 != 3 )
    {
      LODWORD(a5) = 0;
      if ( v16 != 1 )
        return (unsigned int)a5;
      goto LABEL_29;
    }
    if ( (*(_BYTE *)(*v12 + 37LL) & 0xFu) > 1 || (*(_BYTE *)(*v12 + 32LL) & 4) != 0 )
    {
      LOBYTE(a5) = v12 != v14;
      return (unsigned int)a5;
    }
    ++v12;
  }
  if ( (*(_BYTE *)(*v12 + 37LL) & 0xFu) > 1 || (*(_BYTE *)(*v12 + 32LL) & 4) != 0 )
    goto LABEL_20;
  ++v12;
LABEL_29:
  if ( (*(_BYTE *)(*v12 + 37LL) & 0xFu) > 1 || (LODWORD(a5) = 0, (*(_BYTE *)(*v12 + 32LL) & 4) != 0) )
  {
LABEL_20:
    LOBYTE(a5) = v14 != v12;
    return (unsigned int)a5;
  }
  return (unsigned int)a5;
}
