// Function: sub_1F6C3D0
// Address: 0x1f6c3d0
//
__int64 __fastcall sub_1F6C3D0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r8
  int v5; // edx
  _QWORD *v6; // rcx
  __int64 v7; // rdi
  int v8; // r9d
  __int64 v9; // rax
  void **v10; // rax
  __int64 v11; // rax
  __int64 v12; // rcx
  __int64 v13; // rcx
  _QWORD *v14; // rdx
  _QWORD *v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rcx

  v2 = *(_QWORD *)(a1 + 48);
  LODWORD(v3) = 0;
  if ( !v2 || *(_QWORD *)(v2 + 32) )
    return (unsigned int)v3;
  v5 = *(unsigned __int16 *)(a1 + 24);
  if ( ((*(_WORD *)(a1 + 24) - 118) & 0xFFFB) != 0 )
  {
    if ( v5 != 124 )
      return (unsigned int)v3;
    v6 = *(_QWORD **)(a1 + 32);
    v7 = *v6;
    v8 = *(unsigned __int16 *)(*v6 + 24LL);
    if ( ((*(_WORD *)(*v6 + 24LL) - 118) & 0xFFFB) != 0 )
      return (unsigned int)v3;
  }
  else
  {
    v6 = *(_QWORD **)(a1 + 32);
    v7 = *v6;
    v8 = *(unsigned __int16 *)(*v6 + 24LL);
    if ( ((*(_WORD *)(*v6 + 24LL) - 118) & 0xFFFB) != 0 )
    {
      if ( v8 != 124 || v5 != 118 )
        return (unsigned int)v3;
      goto LABEL_8;
    }
    if ( v5 == 118 )
    {
LABEL_8:
      v9 = v6[5];
      LODWORD(v3) = *(unsigned __int16 *)(v9 + 24);
      LOBYTE(v3) = (_DWORD)v3 == 10 || (_DWORD)v3 == 32;
      if ( !(_BYTE)v3 )
        return (unsigned int)v3;
      goto LABEL_9;
    }
  }
  LODWORD(v3) = 0;
  if ( v8 != 118 )
    return (unsigned int)v3;
  v9 = *(_QWORD *)(*(_QWORD *)(v7 + 32) + 40LL);
  LODWORD(v3) = *(unsigned __int16 *)(v9 + 24);
  LOBYTE(v3) = (_DWORD)v3 == 32 || (_DWORD)v3 == 10;
  if ( !(_BYTE)v3 )
    return (unsigned int)v3;
LABEL_9:
  v3 = *(_QWORD *)(v9 + 88);
  v10 = *(void ***)(v3 + 24);
  if ( *(_DWORD *)(v3 + 32) > 0x40u )
    v10 = (void **)*v10;
  if ( v10 == (void **)0xFFFF )
  {
    if ( v5 == 124 )
    {
      v11 = 1;
      goto LABEL_18;
    }
    LOBYTE(v3) = v8 == 122 && v5 == 118;
    if ( !(_BYTE)v3 )
      return (unsigned int)v3;
    v11 = 1;
LABEL_37:
    v16 = *(_QWORD *)(*(_QWORD *)(v7 + 32) + 40LL);
    LOBYTE(v3) = *(_WORD *)(v16 + 24) == 10 || *(_WORD *)(v16 + 24) == 32;
    if ( !(_BYTE)v3 )
      return (unsigned int)v3;
    v17 = *(_QWORD *)(v16 + 88);
    v14 = *(_QWORD **)(v17 + 24);
    if ( *(_DWORD *)(v17 + 32) <= 0x40u )
      goto LABEL_21;
    goto LABEL_20;
  }
  if ( (unsigned __int64)v10 <= 0xFFFF )
  {
    if ( v10 != (void **)255 )
    {
      if ( v10 == (void **)65280 )
      {
        v11 = 1;
        goto LABEL_16;
      }
LABEL_42:
      LODWORD(v3) = 0;
      return (unsigned int)v3;
    }
    v11 = 0;
    if ( v5 != 118 )
    {
      if ( v5 == 122 )
      {
        v11 = 0;
        goto LABEL_18;
      }
      goto LABEL_42;
    }
LABEL_47:
    LODWORD(v3) = 0;
    if ( v8 != 124 )
      return (unsigned int)v3;
    goto LABEL_37;
  }
  if ( v10 == (void **)&loc_FF0000 )
  {
    if ( v5 != 118 )
    {
      if ( v5 == 122 )
      {
        v11 = 2;
        goto LABEL_18;
      }
      goto LABEL_42;
    }
    v11 = 2;
    goto LABEL_47;
  }
  LODWORD(v3) = -16777216;
  if ( v10 != (void **)4278190080LL )
    goto LABEL_42;
  v11 = 3;
LABEL_16:
  if ( v5 == 118 )
  {
    LODWORD(v3) = 0;
    if ( v8 != 122 )
      return (unsigned int)v3;
    goto LABEL_37;
  }
  if ( v5 == 122 )
  {
    LODWORD(v3) = 0;
    return (unsigned int)v3;
  }
LABEL_18:
  v12 = v6[5];
  LOBYTE(v3) = *(_WORD *)(v12 + 24) == 32 || *(_WORD *)(v12 + 24) == 10;
  if ( !(_BYTE)v3 )
    return (unsigned int)v3;
  v13 = *(_QWORD *)(v12 + 88);
  v14 = *(_QWORD **)(v13 + 24);
  if ( *(_DWORD *)(v13 + 32) <= 0x40u )
    goto LABEL_21;
LABEL_20:
  v14 = (_QWORD *)*v14;
LABEL_21:
  LODWORD(v3) = 0;
  if ( v14 == (_QWORD *)8 )
  {
    v15 = (_QWORD *)(a2 + 8 * v11);
    LODWORD(v3) = 0;
    if ( !*v15 )
    {
      LODWORD(v3) = 1;
      *v15 = **(_QWORD **)(v7 + 32);
    }
  }
  return (unsigned int)v3;
}
