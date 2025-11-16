// Function: sub_B58A70
// Address: 0xb58a70
//
__int64 __fastcall sub_B58A70(unsigned __int8 *a1, __int64 a2, __int64 a3)
{
  int v3; // eax
  unsigned int v4; // r12d
  int v6; // eax
  unsigned __int8 *v7; // rdi
  char *v8; // r13
  char v9; // al
  __int64 v10; // rax
  __int64 v11; // r14
  __int64 v12; // rdx
  bool v13; // bl
  __int64 v14; // r13
  unsigned int v15; // ebx
  __int64 v16; // rax
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

  v3 = *a1;
  if ( (unsigned __int8)v3 > 0x1Cu )
  {
    if ( (_BYTE)v3 == 85 )
    {
      a3 = *((_QWORD *)a1 - 4);
      if ( a3 )
      {
        if ( !*(_BYTE *)a3 && *(_QWORD *)(a3 + 24) == *((_QWORD *)a1 + 10) && *(_DWORD *)(a3 + 36) == 493 )
          return 1;
      }
    }
    v6 = v3 - 29;
  }
  else
  {
    if ( (_BYTE)v3 != 5 )
      return 0;
    v6 = *((unsigned __int16 *)a1 + 1);
  }
  if ( v6 != 47 )
    return 0;
  v7 = (a1[7] & 0x40) != 0 ? (unsigned __int8 *)*((_QWORD *)a1 - 1) : &a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
  v8 = *(char **)v7;
  if ( !*(_QWORD *)v7 )
    return 0;
  v9 = *v8;
  if ( (unsigned __int8)*v8 > 0x1Cu )
  {
    if ( v9 != 63 )
      return 0;
LABEL_13:
    v10 = sub_BB5290(*(_QWORD *)v7, a2, a3);
    if ( *(_BYTE *)(v10 + 8) != 18 )
      return 0;
    if ( (*((_DWORD *)v8 + 1) & 0x7FFFFFF) != 2 )
      return 0;
    v4 = sub_BCAC40(*(_QWORD *)(v10 + 24), 8);
    if ( !(_BYTE)v4 )
      return 0;
    v11 = *(_QWORD *)&v8[-32 * (*((_DWORD *)v8 + 1) & 0x7FFFFFF)];
    if ( *(_BYTE *)v11 > 0x15u )
      return 0;
    v13 = sub_AC30F0(*(_QWORD *)&v8[-32 * (*((_DWORD *)v8 + 1) & 0x7FFFFFF)]);
    if ( !v13 )
    {
      if ( *(_BYTE *)v11 == 17 )
      {
        v17 = *(_DWORD *)(v11 + 32);
        if ( v17 <= 0x40 )
          v18 = *(_QWORD *)(v11 + 24) == 0;
        else
          v18 = v17 == (unsigned int)sub_C444A0(v11 + 24);
      }
      else
      {
        v21 = *(_QWORD *)(v11 + 8);
        if ( (unsigned int)*(unsigned __int8 *)(v21 + 8) - 17 > 1 )
          return 0;
        v22 = sub_AD7630(v11, 0, v12);
        if ( !v22 || *v22 != 17 )
        {
          if ( *(_BYTE *)(v21 + 8) != 17 )
            return 0;
          v24 = *(_DWORD *)(v21 + 32);
          v25 = 0;
          v28 = v24;
          while ( v28 != v25 )
          {
            v26 = sub_AD69F0((unsigned __int8 *)v11, v25);
            if ( !v26 )
              return 0;
            if ( *(_BYTE *)v26 != 13 )
            {
              if ( *(_BYTE *)v26 != 17 )
                return 0;
              v27 = *(_DWORD *)(v26 + 32);
              v13 = v27 <= 0x40 ? *(_QWORD *)(v26 + 24) == 0 : v27 == (unsigned int)sub_C444A0(v26 + 24);
              if ( !v13 )
                return 0;
            }
            ++v25;
          }
          if ( !v13 )
            return 0;
          goto LABEL_18;
        }
        v23 = *((_DWORD *)v22 + 8);
        if ( v23 <= 0x40 )
          v18 = *((_QWORD *)v22 + 3) == 0;
        else
          v18 = v23 == (unsigned int)sub_C444A0(v22 + 24);
      }
      if ( !v18 )
        return 0;
    }
LABEL_18:
    v14 = *(_QWORD *)&v8[32 * (1LL - (*((_DWORD *)v8 + 1) & 0x7FFFFFF))];
    if ( *(_BYTE *)v14 != 17 )
    {
      v19 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v14 + 8) + 8LL) - 17;
      if ( (unsigned int)v19 > 1 )
        return 0;
      if ( *(_BYTE *)v14 > 0x15u )
        return 0;
      v20 = sub_AD7630(v14, 0, v19);
      v14 = (__int64)v20;
      if ( !v20 || *v20 != 17 )
        return 0;
    }
    v15 = *(_DWORD *)(v14 + 32);
    if ( v15 > 0x40 )
    {
      if ( v15 - (unsigned int)sub_C444A0(v14 + 24) > 0x40 )
        return 0;
      v16 = **(_QWORD **)(v14 + 24);
    }
    else
    {
      v16 = *(_QWORD *)(v14 + 24);
    }
    if ( v16 == 1 )
      return v4;
    return 0;
  }
  v4 = 0;
  if ( v9 == 5 && *((_WORD *)v8 + 1) == 34 )
    goto LABEL_13;
  return v4;
}
