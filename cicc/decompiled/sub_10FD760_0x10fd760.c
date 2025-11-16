// Function: sub_10FD760
// Address: 0x10fd760
//
__int64 __fastcall sub_10FD760(unsigned __int8 *a1)
{
  int v1; // eax
  unsigned int v2; // r12d
  int v4; // eax
  __int64 *v5; // rdi
  __int64 v6; // r13
  char v7; // al
  __int64 v8; // rax
  unsigned int v9; // eax
  __int64 v10; // r14
  __int64 v11; // rdx
  bool v12; // bl
  __int64 v13; // r13
  unsigned int v14; // ebx
  __int64 v15; // rax
  __int64 v16; // rdx
  unsigned int v17; // ebx
  bool v18; // al
  __int64 v19; // rdx
  _BYTE *v20; // rax
  __int64 v21; // r15
  _BYTE *v22; // rax
  unsigned int v23; // ebx
  int v24; // eax
  unsigned int v25; // r15d
  __int64 v26; // rax
  unsigned int v27; // ebx
  int v28; // [rsp+Ch] [rbp-34h]

  v1 = *a1;
  if ( (unsigned __int8)v1 > 0x1Cu )
  {
    if ( (_BYTE)v1 == 85 )
    {
      v16 = *((_QWORD *)a1 - 4);
      if ( v16 )
      {
        if ( !*(_BYTE *)v16 && *(_QWORD *)(v16 + 24) == *((_QWORD *)a1 + 10) && *(_DWORD *)(v16 + 36) == 493 )
          return 1;
      }
    }
    v4 = v1 - 29;
  }
  else
  {
    if ( (_BYTE)v1 != 5 )
      return 0;
    v4 = *((unsigned __int16 *)a1 + 1);
  }
  if ( v4 != 47 )
    return 0;
  v5 = (a1[7] & 0x40) != 0 ? (__int64 *)*((_QWORD *)a1 - 1) : (__int64 *)&a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
  v6 = *v5;
  if ( !*v5 )
    return 0;
  v7 = *(_BYTE *)v6;
  if ( *(_BYTE *)v6 > 0x1Cu )
  {
    if ( v7 != 63 )
      return 0;
LABEL_13:
    v8 = sub_BB5290(*v5);
    if ( *(_BYTE *)(v8 + 8) != 18 )
      return 0;
    if ( (*(_DWORD *)(v6 + 4) & 0x7FFFFFF) != 2 )
      return 0;
    LOBYTE(v9) = sub_BCAC40(*(_QWORD *)(v8 + 24), 8);
    v2 = v9;
    if ( !(_BYTE)v9 )
      return 0;
    v10 = *(_QWORD *)(v6 - 32LL * (*(_DWORD *)(v6 + 4) & 0x7FFFFFF));
    if ( *(_BYTE *)v10 > 0x15u )
      return 0;
    v12 = sub_AC30F0(*(_QWORD *)(v6 - 32LL * (*(_DWORD *)(v6 + 4) & 0x7FFFFFF)));
    if ( !v12 )
    {
      if ( *(_BYTE *)v10 == 17 )
      {
        v17 = *(_DWORD *)(v10 + 32);
        if ( v17 <= 0x40 )
          v18 = *(_QWORD *)(v10 + 24) == 0;
        else
          v18 = v17 == (unsigned int)sub_C444A0(v10 + 24);
      }
      else
      {
        v21 = *(_QWORD *)(v10 + 8);
        if ( (unsigned int)*(unsigned __int8 *)(v21 + 8) - 17 > 1 )
          return 0;
        v22 = sub_AD7630(v10, 0, v11);
        if ( !v22 || *v22 != 17 )
        {
          if ( *(_BYTE *)(v21 + 8) != 17 )
            return 0;
          v24 = *(_DWORD *)(v21 + 32);
          v25 = 0;
          v28 = v24;
          while ( v28 != v25 )
          {
            v26 = sub_AD69F0((unsigned __int8 *)v10, v25);
            if ( !v26 )
              return 0;
            if ( *(_BYTE *)v26 != 13 )
            {
              if ( *(_BYTE *)v26 != 17 )
                return 0;
              v27 = *(_DWORD *)(v26 + 32);
              v12 = v27 <= 0x40 ? *(_QWORD *)(v26 + 24) == 0 : v27 == (unsigned int)sub_C444A0(v26 + 24);
              if ( !v12 )
                return 0;
            }
            ++v25;
          }
          if ( !v12 )
            return 0;
          goto LABEL_18;
        }
        v23 = *((_DWORD *)v22 + 8);
        if ( v23 <= 0x40 )
          v18 = *((_QWORD *)v22 + 3) == 0;
        else
          v18 = v23 == (unsigned int)sub_C444A0((__int64)(v22 + 24));
      }
      if ( !v18 )
        return 0;
    }
LABEL_18:
    v13 = *(_QWORD *)(v6 + 32 * (1LL - (*(_DWORD *)(v6 + 4) & 0x7FFFFFF)));
    if ( *(_BYTE *)v13 != 17 )
    {
      v19 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v13 + 8) + 8LL) - 17;
      if ( (unsigned int)v19 > 1 )
        return 0;
      if ( *(_BYTE *)v13 > 0x15u )
        return 0;
      v20 = sub_AD7630(v13, 0, v19);
      v13 = (__int64)v20;
      if ( !v20 || *v20 != 17 )
        return 0;
    }
    v14 = *(_DWORD *)(v13 + 32);
    if ( v14 > 0x40 )
    {
      if ( v14 - (unsigned int)sub_C444A0(v13 + 24) > 0x40 )
        return 0;
      v15 = **(_QWORD **)(v13 + 24);
    }
    else
    {
      v15 = *(_QWORD *)(v13 + 24);
    }
    if ( v15 == 1 )
      return v2;
    return 0;
  }
  v2 = 0;
  if ( v7 == 5 && *(_WORD *)(v6 + 2) == 34 )
    goto LABEL_13;
  return v2;
}
