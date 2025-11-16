// Function: sub_325E3E0
// Address: 0x325e3e0
//
__int64 __fastcall sub_325E3E0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned int v3; // r8d
  int v5; // ecx
  _QWORD *v6; // rdi
  __int64 v7; // r9
  int v8; // r10d
  __int64 v9; // rax
  __int64 v10; // rdx
  void **v11; // rax
  __int64 v12; // rax
  __int64 v13; // rcx
  __int64 v14; // rcx
  _QWORD *v15; // rdx
  _QWORD *v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rcx

  v2 = *(_QWORD *)(a1 + 56);
  v3 = 0;
  if ( !v2 || *(_QWORD *)(v2 + 32) )
    return v3;
  v5 = *(_DWORD *)(a1 + 24);
  if ( (v5 & 0xFFFFFFFB) == 0xBA )
  {
    v6 = *(_QWORD **)(a1 + 40);
    v7 = *v6;
    v8 = *(_DWORD *)(*v6 + 24LL);
    if ( (v8 & 0xFFFFFFFB) != 0xBA )
    {
      LOBYTE(v3) = v5 == 186 && v8 == 192;
      if ( !(_BYTE)v3 )
        return v3;
      v8 = 192;
      goto LABEL_8;
    }
    if ( v5 == 186 )
    {
LABEL_8:
      v9 = v6[5];
      LOBYTE(v3) = *(_DWORD *)(v9 + 24) == 35 || *(_DWORD *)(v9 + 24) == 11;
      if ( !(_BYTE)v3 )
        return v3;
      goto LABEL_9;
    }
  }
  else
  {
    if ( v5 != 192 )
      return v3;
    v6 = *(_QWORD **)(a1 + 40);
    v7 = *v6;
    v8 = *(_DWORD *)(*v6 + 24LL);
    if ( (v8 & 0xFFFFFFFB) != 0xBA )
      return v3;
  }
  v3 = 0;
  if ( v8 != 186 )
    return v3;
  v9 = *(_QWORD *)(*(_QWORD *)(v7 + 40) + 40LL);
  LOBYTE(v3) = *(_DWORD *)(v9 + 24) == 11 || *(_DWORD *)(v9 + 24) == 35;
  if ( !(_BYTE)v3 )
    return v3;
LABEL_9:
  v10 = *(_QWORD *)(v9 + 96);
  v11 = *(void ***)(v10 + 24);
  if ( *(_DWORD *)(v10 + 32) > 0x40u )
    v11 = (void **)*v11;
  if ( v11 == (void **)0xFFFF )
  {
    if ( v5 == 192 )
    {
      v12 = 1;
      goto LABEL_18;
    }
    LOBYTE(v3) = v8 == 190 && v5 == 186;
    if ( !(_BYTE)v3 )
      return v3;
    v12 = 1;
LABEL_37:
    v17 = *(_QWORD *)(*(_QWORD *)(v7 + 40) + 40LL);
    LOBYTE(v3) = *(_DWORD *)(v17 + 24) == 11 || *(_DWORD *)(v17 + 24) == 35;
    if ( !(_BYTE)v3 )
      return v3;
    v18 = *(_QWORD *)(v17 + 96);
    v15 = *(_QWORD **)(v18 + 24);
    if ( *(_DWORD *)(v18 + 32) <= 0x40u )
      goto LABEL_21;
    goto LABEL_20;
  }
  if ( (unsigned __int64)v11 <= 0xFFFF )
  {
    if ( v11 != (void **)255 )
    {
      if ( v11 == (void **)65280 )
      {
        v12 = 1;
        goto LABEL_16;
      }
      return 0;
    }
    v12 = 0;
    if ( v5 != 186 )
    {
      if ( v5 == 190 )
      {
        v12 = 0;
        goto LABEL_18;
      }
      return 0;
    }
LABEL_47:
    v3 = 0;
    if ( v8 != 192 )
      return v3;
    goto LABEL_37;
  }
  if ( v11 == (void **)&loc_FF0000 )
  {
    if ( v5 != 186 )
    {
      if ( v5 == 190 )
      {
        v12 = 2;
        goto LABEL_18;
      }
      return 0;
    }
    v12 = 2;
    goto LABEL_47;
  }
  if ( v11 != (void **)4278190080LL )
    return 0;
  v12 = 3;
LABEL_16:
  if ( v5 == 186 )
  {
    v3 = 0;
    if ( v8 != 190 )
      return v3;
    goto LABEL_37;
  }
  if ( v5 == 190 )
    return 0;
LABEL_18:
  v13 = v6[5];
  LOBYTE(v3) = *(_DWORD *)(v13 + 24) == 35 || *(_DWORD *)(v13 + 24) == 11;
  if ( !(_BYTE)v3 )
    return v3;
  v14 = *(_QWORD *)(v13 + 96);
  v15 = *(_QWORD **)(v14 + 24);
  if ( *(_DWORD *)(v14 + 32) <= 0x40u )
    goto LABEL_21;
LABEL_20:
  v15 = (_QWORD *)*v15;
LABEL_21:
  v3 = 0;
  if ( v15 == (_QWORD *)8 )
  {
    v16 = (_QWORD *)(a2 + 8 * v12);
    v3 = 0;
    if ( !*v16 )
    {
      v3 = 1;
      *v16 = **(_QWORD **)(v7 + 40);
    }
  }
  return v3;
}
