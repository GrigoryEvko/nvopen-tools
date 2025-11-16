// Function: sub_1181510
// Address: 0x1181510
//
__int64 __fastcall sub_1181510(_QWORD **a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // r13
  unsigned int v6; // r14d
  bool v7; // al
  __int64 *v8; // rax
  __int64 v9; // rbx
  __int64 v10; // rax
  __int64 v11; // r14
  __int64 v12; // rdx
  _BYTE *v13; // rax
  unsigned int v14; // r14d
  char v15; // r15
  unsigned int v16; // r14d
  __int64 v17; // rax
  unsigned int v18; // r15d
  int v19; // [rsp-3Ch] [rbp-3Ch]

  if ( *(_BYTE *)a2 != 86 )
    return 0;
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
  {
    v3 = **(_QWORD **)(a2 - 8);
    if ( !v3 )
      return 0;
  }
  else
  {
    v3 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
    if ( !v3 )
      return 0;
  }
  **a1 = v3;
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v4 = *(_QWORD *)(a2 - 8);
  else
    v4 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  v5 = *(_QWORD *)(v4 + 32);
  if ( *(_BYTE *)v5 == 17 )
  {
    v6 = *(_DWORD *)(v5 + 32);
    if ( v6 <= 0x40 )
      v7 = *(_QWORD *)(v5 + 24) == 1;
    else
      v7 = v6 - 1 == (unsigned int)sub_C444A0(v5 + 24);
LABEL_10:
    if ( !v7 )
      return 0;
    goto LABEL_11;
  }
  v11 = *(_QWORD *)(v5 + 8);
  v12 = (unsigned int)*(unsigned __int8 *)(v11 + 8) - 17;
  if ( (unsigned int)v12 > 1 || *(_BYTE *)v5 > 0x15u )
    return 0;
  v13 = sub_AD7630(v5, 0, v12);
  if ( !v13 || *v13 != 17 )
  {
    if ( *(_BYTE *)(v11 + 8) == 17 )
    {
      v19 = *(_DWORD *)(v11 + 32);
      if ( v19 )
      {
        v15 = 0;
        v16 = 0;
        while ( 1 )
        {
          v17 = sub_AD69F0((unsigned __int8 *)v5, v16);
          if ( !v17 )
            break;
          if ( *(_BYTE *)v17 != 13 )
          {
            if ( *(_BYTE *)v17 != 17 )
              return 0;
            v18 = *(_DWORD *)(v17 + 32);
            if ( v18 <= 0x40 )
            {
              if ( *(_QWORD *)(v17 + 24) != 1 )
                return 0;
            }
            else if ( (unsigned int)sub_C444A0(v17 + 24) != v18 - 1 )
            {
              return 0;
            }
            v15 = 1;
          }
          if ( v19 == ++v16 )
          {
            if ( v15 )
              goto LABEL_11;
            return 0;
          }
        }
      }
    }
    return 0;
  }
  v14 = *((_DWORD *)v13 + 8);
  if ( v14 > 0x40 )
  {
    v7 = v14 - 1 == (unsigned int)sub_C444A0((__int64)(v13 + 24));
    goto LABEL_10;
  }
  if ( *((_QWORD *)v13 + 3) != 1 )
    return 0;
LABEL_11:
  v8 = a1[1];
  if ( v8 )
    *v8 = v5;
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v9 = *(_QWORD *)(a2 - 8);
  else
    v9 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  v10 = *(_QWORD *)(v9 + 64);
  if ( v10 )
  {
    *a1[2] = v10;
    return 1;
  }
  return 0;
}
