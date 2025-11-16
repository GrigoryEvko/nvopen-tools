// Function: sub_2DA2530
// Address: 0x2da2530
//
__int64 __fastcall sub_2DA2530(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rcx
  _BYTE *v7; // r13
  __int64 v8; // r14
  unsigned int v9; // r15d
  bool v10; // al
  _QWORD *v11; // rax
  _BYTE *v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rdx
  _BYTE *v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // r15
  __int64 v19; // rdx
  _BYTE *v20; // rax
  unsigned int v21; // r15d
  int v22; // eax
  unsigned int v23; // edx
  char v24; // r15
  __int64 v25; // rax
  unsigned int v26; // edx
  unsigned int v27; // r15d
  unsigned int v28; // [rsp-40h] [rbp-40h]
  int v29; // [rsp-3Ch] [rbp-3Ch]

  if ( *(_BYTE *)a2 != 85 )
    return 0;
  v2 = *(_QWORD *)(a2 - 32);
  if ( !v2 )
    return 0;
  if ( *(_BYTE *)v2 )
    return 0;
  if ( *(_QWORD *)(v2 + 24) != *(_QWORD *)(a2 + 80) )
    return 0;
  if ( *(_DWORD *)(v2 + 36) != *(_DWORD *)a1 )
    return 0;
  v4 = *(_QWORD *)(a2 + 32 * (*(unsigned int *)(a1 + 8) - (unsigned __int64)(*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
  if ( *(_BYTE *)v4 != 85 )
    return 0;
  v5 = *(_QWORD *)(v4 - 32);
  if ( !v5 )
    return 0;
  if ( *(_BYTE *)v5 )
    return 0;
  if ( *(_QWORD *)(v5 + 24) != *(_QWORD *)(v4 + 80) )
    return 0;
  if ( *(_DWORD *)(v5 + 36) != *(_DWORD *)(a1 + 16) )
    return 0;
  v6 = *(_QWORD *)(v4 + 32 * (*(unsigned int *)(a1 + 24) - (unsigned __int64)(*(_DWORD *)(v4 + 4) & 0x7FFFFFF)));
  if ( !v6 )
    return 0;
  **(_QWORD **)(a1 + 32) = v6;
  if ( *(_BYTE *)v4 != 85 )
    return 0;
  v7 = *(_BYTE **)(v4 + 32 * (*(unsigned int *)(a1 + 40) - (unsigned __int64)(*(_DWORD *)(v4 + 4) & 0x7FFFFFF)));
  if ( *v7 != 44 )
    return 0;
  v8 = *((_QWORD *)v7 - 8);
  if ( *(_BYTE *)v8 == 17 )
  {
    v9 = *(_DWORD *)(v8 + 32);
    if ( v9 <= 0x40 )
      v10 = *(_QWORD *)(v8 + 24) == 0;
    else
      v10 = v9 == (unsigned int)sub_C444A0(v8 + 24);
    if ( v10 )
      goto LABEL_20;
    return 0;
  }
  v18 = *(_QWORD *)(v8 + 8);
  v19 = (unsigned int)*(unsigned __int8 *)(v18 + 8) - 17;
  if ( (unsigned int)v19 > 1 || *(_BYTE *)v8 > 0x15u )
    return 0;
  v20 = sub_AD7630(*((_QWORD *)v7 - 8), 0, v19);
  if ( v20 && *v20 == 17 )
  {
    v21 = *((_DWORD *)v20 + 8);
    if ( v21 <= 0x40 )
    {
      if ( *((_QWORD *)v20 + 3) )
        return 0;
    }
    else if ( v21 != (unsigned int)sub_C444A0((__int64)(v20 + 24)) )
    {
      return 0;
    }
  }
  else
  {
    if ( *(_BYTE *)(v18 + 8) != 17 )
      return 0;
    v22 = *(_DWORD *)(v18 + 32);
    v23 = 0;
    v24 = 0;
    v29 = v22;
    while ( v29 != v23 )
    {
      v28 = v23;
      v25 = sub_AD69F0((unsigned __int8 *)v8, v23);
      if ( !v25 )
        return 0;
      v26 = v28;
      if ( *(_BYTE *)v25 != 13 )
      {
        if ( *(_BYTE *)v25 != 17 )
          return 0;
        v27 = *(_DWORD *)(v25 + 32);
        if ( v27 <= 0x40 )
        {
          if ( *(_QWORD *)(v25 + 24) )
            return 0;
          v24 = 1;
        }
        else
        {
          if ( v27 != (unsigned int)sub_C444A0(v25 + 24) )
            return 0;
          v26 = v28;
          v24 = 1;
        }
      }
      v23 = v26 + 1;
    }
    if ( !v24 )
      return 0;
  }
LABEL_20:
  v11 = *(_QWORD **)(a1 + 48);
  if ( v11 )
    *v11 = v8;
  v12 = (_BYTE *)*((_QWORD *)v7 - 4);
  if ( *v12 != 46 )
    return 0;
  v13 = *((_QWORD *)v12 - 8);
  if ( !v13 )
    return 0;
  **(_QWORD **)(a1 + 56) = v13;
  v14 = *((_QWORD *)v12 - 4);
  if ( !v14 )
    return 0;
  **(_QWORD **)(a1 + 64) = v14;
  if ( *(_BYTE *)a2 != 85 )
    return 0;
  v15 = *(_BYTE **)(a2 + 32 * (*(unsigned int *)(a1 + 72) - (unsigned __int64)(*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
  if ( *v15 != 46 )
    return 0;
  v16 = *((_QWORD *)v15 - 8);
  if ( !v16 )
    return 0;
  **(_QWORD **)(a1 + 80) = v16;
  v17 = *((_QWORD *)v15 - 4);
  if ( !v17 )
    return 0;
  **(_QWORD **)(a1 + 88) = v17;
  return 1;
}
