// Function: sub_11581D0
// Address: 0x11581d0
//
__int64 __fastcall sub_11581D0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // r12
  unsigned int v7; // r13d
  bool v8; // al
  __int64 *v9; // rax
  __int64 v10; // r13
  __int64 v11; // rdx
  _BYTE *v12; // rax
  unsigned int v13; // r13d
  int v14; // r13d
  unsigned int v15; // r14d
  char v16; // r15
  __int64 v17; // rax
  unsigned int v18; // r15d

  v2 = *(_QWORD *)(a2 + 16);
  if ( !v2 )
    return 0;
  if ( *(_QWORD *)(v2 + 8) )
    return 0;
  if ( *(_BYTE *)a2 != 85 )
    return 0;
  v4 = *(_QWORD *)(a2 - 32);
  if ( !v4 || *(_BYTE *)v4 || *(_QWORD *)(v4 + 24) != *(_QWORD *)(a2 + 80) )
    return 0;
  if ( *(_DWORD *)(v4 + 36) != *(_DWORD *)a1 )
    return 0;
  v5 = *(_QWORD *)(a2 + 32 * (*(unsigned int *)(a1 + 8) - (unsigned __int64)(*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
  if ( !v5 )
    return 0;
  **(_QWORD **)(a1 + 16) = v5;
  if ( *(_BYTE *)a2 != 85 )
    return 0;
  v6 = *(_QWORD *)(a2 + 32 * (*(unsigned int *)(a1 + 24) - (unsigned __int64)(*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
  if ( *(_BYTE *)v6 == 17 )
  {
    v7 = *(_DWORD *)(v6 + 32);
    if ( v7 <= 0x40 )
      v8 = *(_QWORD *)(v6 + 24) == 1;
    else
      v8 = v7 - 1 == (unsigned int)sub_C444A0(v6 + 24);
    if ( v8 )
      goto LABEL_16;
    return 0;
  }
  v10 = *(_QWORD *)(v6 + 8);
  v11 = (unsigned int)*(unsigned __int8 *)(v10 + 8) - 17;
  if ( (unsigned int)v11 > 1 || *(_BYTE *)v6 > 0x15u )
    return 0;
  v12 = sub_AD7630(v6, 0, v11);
  if ( !v12 || *v12 != 17 )
  {
    if ( *(_BYTE *)(v10 + 8) == 17 )
    {
      v14 = *(_DWORD *)(v10 + 32);
      v15 = 0;
      v16 = 0;
      while ( v14 != v15 )
      {
        v17 = sub_AD69F0((unsigned __int8 *)v6, v15);
        if ( !v17 )
          return 0;
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
          v16 = 1;
        }
        ++v15;
      }
      if ( v16 )
        goto LABEL_16;
    }
    return 0;
  }
  v13 = *((_DWORD *)v12 + 8);
  if ( v13 <= 0x40 )
  {
    if ( *((_QWORD *)v12 + 3) != 1 )
      return 0;
  }
  else if ( (unsigned int)sub_C444A0((__int64)(v12 + 24)) != v13 - 1 )
  {
    return 0;
  }
LABEL_16:
  v9 = *(__int64 **)(a1 + 32);
  if ( v9 )
    *v9 = v6;
  return 1;
}
