// Function: sub_100C010
// Address: 0x100c010
//
__int64 __fastcall sub_100C010(__int64 a1, int a2, unsigned __int8 *a3)
{
  unsigned int v3; // r12d
  char v7; // al
  __int64 v8; // r14
  __int64 v9; // rdi
  unsigned int v10; // eax
  __int64 v11; // r15
  _BYTE *v12; // rdi
  __int64 v13; // r14
  __int64 v14; // rax
  unsigned __int8 *v15; // rbx
  __int64 v16; // rdi
  __int64 v17; // rax
  __int64 v18; // r12
  _BYTE *v19; // rdi
  __int64 v20; // rbx
  int v21; // eax
  __int64 v22; // rdx
  unsigned __int8 *v23; // rbx
  __int64 v24; // rdx
  _QWORD *v25; // r14
  __int64 v26; // rax

  if ( a2 + 29 != *a3 )
    return 0;
  v7 = sub_995B10((_QWORD **)a1, *((_QWORD *)a3 - 8));
  v8 = *((_QWORD *)a3 - 4);
  if ( v7 && *(_BYTE *)v8 > 0x1Cu )
  {
    v9 = *(_QWORD *)(v8 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v9 + 8) - 17 <= 1 )
      v9 = **(_QWORD **)(v9 + 16);
    LOBYTE(v10) = sub_BCAC40(v9, 1);
    v3 = v10;
    if ( (_BYTE)v10 )
    {
      if ( *(_BYTE *)v8 == 57 )
      {
        if ( (*(_BYTE *)(v8 + 7) & 0x40) != 0 )
          v25 = *(_QWORD **)(v8 - 8);
        else
          v25 = (_QWORD *)(v8 - 32LL * (*(_DWORD *)(v8 + 4) & 0x7FFFFFF));
        v26 = *(_QWORD *)(a1 + 8);
        if ( *v25 == v26 || v25[4] == v26 )
          return v3;
      }
      else if ( *(_BYTE *)v8 == 86 )
      {
        v11 = *(_QWORD *)(v8 - 96);
        if ( *(_QWORD *)(v11 + 8) == *(_QWORD *)(v8 + 8) )
        {
          v12 = *(_BYTE **)(v8 - 32);
          if ( *v12 <= 0x15u )
          {
            v13 = *(_QWORD *)(v8 - 64);
            if ( sub_AC30F0((__int64)v12) )
            {
              v14 = *(_QWORD *)(a1 + 8);
              if ( v11 == v14 || v13 == v14 )
                return v3;
            }
          }
        }
      }
    }
    v8 = *((_QWORD *)a3 - 4);
  }
  if ( !(unsigned __int8)sub_995B10((_QWORD **)a1, v8) )
    return 0;
  v15 = (unsigned __int8 *)*((_QWORD *)a3 - 8);
  if ( *v15 <= 0x1Cu )
    return 0;
  v16 = *((_QWORD *)v15 + 1);
  if ( (unsigned int)*(unsigned __int8 *)(v16 + 8) - 17 <= 1 )
    v16 = **(_QWORD **)(v16 + 16);
  if ( !sub_BCAC40(v16, 1) )
    return 0;
  LODWORD(v17) = *v15;
  if ( (_BYTE)v17 != 57 )
  {
    if ( (_BYTE)v17 == 86 )
    {
      v18 = *((_QWORD *)v15 - 12);
      if ( *(_QWORD *)(v18 + 8) == *((_QWORD *)v15 + 1) )
      {
        v19 = (_BYTE *)*((_QWORD *)v15 - 4);
        if ( *v19 <= 0x15u )
        {
          v20 = *((_QWORD *)v15 - 8);
          if ( sub_AC30F0((__int64)v19) )
          {
            v22 = *(_QWORD *)(a1 + 8);
            LOBYTE(v18) = v18 == v22;
            LOBYTE(v21) = v20 == v22;
            return v21 | (unsigned int)v18;
          }
        }
      }
    }
    return 0;
  }
  if ( (v15[7] & 0x40) != 0 )
  {
    v23 = (unsigned __int8 *)*((_QWORD *)v15 - 1);
  }
  else
  {
    v17 = 32LL * (*((_DWORD *)v15 + 1) & 0x7FFFFFF);
    v23 = &v15[-v17];
  }
  v24 = *(_QWORD *)(a1 + 8);
  LOBYTE(v3) = *(_QWORD *)v23 == v24;
  LOBYTE(v17) = *((_QWORD *)v23 + 4) == v24;
  v3 |= v17;
  return v3;
}
