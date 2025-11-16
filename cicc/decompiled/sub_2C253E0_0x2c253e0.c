// Function: sub_2C253E0
// Address: 0x2c253e0
//
__int64 __fastcall sub_2C253E0(__int64 a1)
{
  unsigned int v1; // r8d
  __int64 v2; // rax
  __int64 v3; // rdx
  char v4; // al
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  _QWORD *v8; // rax
  __int64 v9; // rdx
  _QWORD *v10; // rbx
  __int64 v11; // rcx
  __int64 v12; // rdx
  _QWORD *v13; // rdx

  if ( *(_BYTE *)(a1 + 8) == 9 )
  {
    v1 = *(unsigned __int8 *)(a1 + 161);
    if ( (_BYTE)v1 )
    {
      v2 = *(_QWORD *)(a1 + 136);
      if ( *(_BYTE *)v2 == 85 )
      {
        v3 = *(_QWORD *)(v2 - 32);
        if ( v3 )
        {
          if ( !*(_BYTE *)v3 && *(_QWORD *)(v3 + 24) == *(_QWORD *)(v2 + 80) && *(_DWORD *)(v3 + 36) == 11 )
            return v1;
        }
      }
    }
  }
  v4 = sub_2C1AB20(a1);
  v1 = 0;
  if ( v4 )
    return v1;
  v6 = *(_QWORD *)(a1 + 16);
  v1 = 1;
  v7 = v6 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (v6 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    return v1;
  if ( (v6 & 4) == 0 )
  {
    v8 = (_QWORD *)(a1 + 16);
    v10 = (_QWORD *)(a1 + 24);
    goto LABEL_20;
  }
  v8 = *(_QWORD **)v7;
  v9 = 8LL * *(unsigned int *)(v7 + 8);
  v10 = &v8[(unsigned __int64)v9 / 8];
  v11 = v9 >> 3;
  v12 = v9 >> 5;
  if ( v12 )
  {
    v13 = &v8[4 * v12];
    while ( !*(_DWORD *)(*v8 + 24LL) )
    {
      if ( *(_DWORD *)(v8[1] + 24LL) )
      {
        LOBYTE(v1) = v10 == v8 + 1;
        return v1;
      }
      v1 = *(_DWORD *)(v8[2] + 24LL);
      if ( v1 )
      {
        LOBYTE(v1) = v10 == v8 + 2;
        return v1;
      }
      if ( *(_DWORD *)(v8[3] + 24LL) )
      {
        LOBYTE(v1) = v10 == v8 + 3;
        return v1;
      }
      v8 += 4;
      if ( v13 == v8 )
      {
        v11 = v10 - v8;
        goto LABEL_26;
      }
    }
    goto LABEL_18;
  }
LABEL_26:
  if ( v11 == 2 )
    goto LABEL_30;
  if ( v11 != 3 )
  {
    if ( v11 != 1 )
      return 1;
    goto LABEL_20;
  }
  if ( !*(_DWORD *)(*v8 + 24LL) )
  {
    ++v8;
LABEL_30:
    if ( *(_DWORD *)(*v8 + 24LL) )
      goto LABEL_21;
    ++v8;
LABEL_20:
    v1 = 1;
    if ( !*(_DWORD *)(*v8 + 24LL) )
      return v1;
LABEL_21:
    LOBYTE(v1) = v10 == v8;
    return v1;
  }
LABEL_18:
  LOBYTE(v1) = v8 == v10;
  return v1;
}
