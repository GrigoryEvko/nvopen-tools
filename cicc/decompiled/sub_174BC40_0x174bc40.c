// Function: sub_174BC40
// Address: 0x174bc40
//
__int64 __fastcall sub_174BC40(_QWORD **a1, __int64 a2)
{
  char v2; // al
  __int64 v4; // rax
  char v5; // dl
  __int64 v6; // rcx
  int v7; // edx
  int v8; // edx
  __int64 *v9; // rcx
  __int64 v10; // rcx
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  char v14; // dl
  __int64 v15; // rcx
  int v16; // edx
  int v17; // edx
  __int64 *v18; // rcx
  __int64 v19; // rcx
  __int64 v20; // rax
  __int64 v21; // rcx
  int v22; // edx
  int v23; // edx
  __int64 *v24; // rcx
  __int64 v25; // rcx
  __int64 v26; // rcx
  int v27; // edx
  int v28; // edx
  __int64 *v29; // rcx
  __int64 v30; // rcx

  v2 = *(_BYTE *)(a2 + 16);
  if ( v2 != 49 )
  {
    if ( v2 != 5 || *(_WORD *)(a2 + 18) != 25 )
      return 0;
    v13 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
    v14 = *(_BYTE *)(v13 + 16);
    if ( v14 == 47 )
    {
      v26 = *(_QWORD *)(v13 - 48);
      v27 = *(unsigned __int8 *)(v26 + 16);
      if ( (unsigned __int8)v27 > 0x17u )
      {
        v28 = v27 - 24;
      }
      else
      {
        if ( (_BYTE)v27 != 5 )
          return 0;
        v28 = *(unsigned __int16 *)(v26 + 18);
      }
      if ( v28 != 36 )
        return 0;
      v29 = (*(_BYTE *)(v26 + 23) & 0x40) != 0
          ? *(__int64 **)(v26 - 8)
          : (__int64 *)(v26 - 24LL * (*(_DWORD *)(v26 + 20) & 0xFFFFFFF));
      v30 = *v29;
      if ( !v30 )
        return 0;
      **a1 = v30;
      v20 = *(_QWORD *)(v13 - 24);
      if ( *(_BYTE *)(v20 + 16) != 13 )
        return 0;
    }
    else
    {
      if ( v14 != 5 || *(_WORD *)(v13 + 18) != 23 )
        return 0;
      v15 = *(_QWORD *)(v13 - 24LL * (*(_DWORD *)(v13 + 20) & 0xFFFFFFF));
      v16 = *(unsigned __int8 *)(v15 + 16);
      if ( (unsigned __int8)v16 > 0x17u )
      {
        v17 = v16 - 24;
      }
      else
      {
        if ( (_BYTE)v16 != 5 )
          return 0;
        v17 = *(unsigned __int16 *)(v15 + 18);
      }
      if ( v17 != 36 )
        return 0;
      v18 = (*(_BYTE *)(v15 + 23) & 0x40) != 0
          ? *(__int64 **)(v15 - 8)
          : (__int64 *)(v15 - 24LL * (*(_DWORD *)(v15 + 20) & 0xFFFFFFF));
      v19 = *v18;
      if ( !v19 )
        return 0;
      **a1 = v19;
      v20 = *(_QWORD *)(v13 + 24 * (1LL - (*(_DWORD *)(v13 + 20) & 0xFFFFFFF)));
      if ( *(_BYTE *)(v20 + 16) != 13 )
        return 0;
    }
    *a1[1] = v20;
    v12 = *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
    if ( *(_BYTE *)(v12 + 16) == 13 )
      goto LABEL_30;
    return 0;
  }
  v4 = *(_QWORD *)(a2 - 48);
  v5 = *(_BYTE *)(v4 + 16);
  if ( v5 == 47 )
  {
    v21 = *(_QWORD *)(v4 - 48);
    v22 = *(unsigned __int8 *)(v21 + 16);
    if ( (unsigned __int8)v22 > 0x17u )
    {
      v23 = v22 - 24;
    }
    else
    {
      if ( (_BYTE)v22 != 5 )
        return 0;
      v23 = *(unsigned __int16 *)(v21 + 18);
    }
    if ( v23 != 36 )
      return 0;
    v24 = (*(_BYTE *)(v21 + 23) & 0x40) != 0
        ? *(__int64 **)(v21 - 8)
        : (__int64 *)(v21 - 24LL * (*(_DWORD *)(v21 + 20) & 0xFFFFFFF));
    v25 = *v24;
    if ( !v25 )
      return 0;
    **a1 = v25;
    v11 = *(_QWORD *)(v4 - 24);
    if ( *(_BYTE *)(v11 + 16) != 13 )
      return 0;
  }
  else
  {
    if ( v5 != 5 || *(_WORD *)(v4 + 18) != 23 )
      return 0;
    v6 = *(_QWORD *)(v4 - 24LL * (*(_DWORD *)(v4 + 20) & 0xFFFFFFF));
    v7 = *(unsigned __int8 *)(v6 + 16);
    if ( (unsigned __int8)v7 > 0x17u )
    {
      v8 = v7 - 24;
    }
    else
    {
      if ( (_BYTE)v7 != 5 )
        return 0;
      v8 = *(unsigned __int16 *)(v6 + 18);
    }
    if ( v8 != 36 )
      return 0;
    v9 = (*(_BYTE *)(v6 + 23) & 0x40) != 0
       ? *(__int64 **)(v6 - 8)
       : (__int64 *)(v6 - 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF));
    v10 = *v9;
    if ( !v10 )
      return 0;
    **a1 = v10;
    v11 = *(_QWORD *)(v4 + 24 * (1LL - (*(_DWORD *)(v4 + 20) & 0xFFFFFFF)));
    if ( *(_BYTE *)(v11 + 16) != 13 )
      return 0;
  }
  *a1[1] = v11;
  v12 = *(_QWORD *)(a2 - 24);
  if ( *(_BYTE *)(v12 + 16) != 13 )
    return 0;
LABEL_30:
  *a1[2] = v12;
  return 1;
}
