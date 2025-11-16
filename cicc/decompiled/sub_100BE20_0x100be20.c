// Function: sub_100BE20
// Address: 0x100be20
//
__int64 __fastcall sub_100BE20(__int64 a1, int a2, unsigned __int8 *a3)
{
  unsigned int v3; // r12d
  char v7; // al
  __int64 v8; // r14
  __int64 v9; // rdi
  unsigned int v10; // eax
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r15
  _BYTE *v15; // rdi
  __int64 v16; // r14
  __int64 v17; // rax
  unsigned __int8 *v18; // rbx
  __int64 v19; // rdi
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // rax
  __int64 v24; // r12
  _BYTE *v25; // rdi
  __int64 v26; // rbx
  int v27; // eax
  __int64 v28; // rdx
  unsigned __int8 *v29; // rbx
  __int64 v30; // rdx
  _QWORD *v31; // r14
  __int64 v32; // rax

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
      if ( *(_BYTE *)v8 == 58 )
      {
        if ( (*(_BYTE *)(v8 + 7) & 0x40) != 0 )
          v31 = *(_QWORD **)(v8 - 8);
        else
          v31 = (_QWORD *)(v8 - 32LL * (*(_DWORD *)(v8 + 4) & 0x7FFFFFF));
        v32 = *(_QWORD *)(a1 + 8);
        if ( *v31 == v32 || v31[4] == v32 )
          return v3;
      }
      else if ( *(_BYTE *)v8 == 86 )
      {
        v14 = *(_QWORD *)(v8 - 96);
        if ( *(_QWORD *)(v14 + 8) == *(_QWORD *)(v8 + 8) )
        {
          v15 = *(_BYTE **)(v8 - 64);
          if ( *v15 <= 0x15u )
          {
            v16 = *(_QWORD *)(v8 - 32);
            if ( sub_AD7A80(v15, 1, v11, v12, v13) )
            {
              v17 = *(_QWORD *)(a1 + 8);
              if ( v14 == v17 || v16 == v17 )
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
  v18 = (unsigned __int8 *)*((_QWORD *)a3 - 8);
  if ( *v18 <= 0x1Cu )
    return 0;
  v19 = *((_QWORD *)v18 + 1);
  if ( (unsigned int)*(unsigned __int8 *)(v19 + 8) - 17 <= 1 )
    v19 = **(_QWORD **)(v19 + 16);
  if ( !sub_BCAC40(v19, 1) )
    return 0;
  LODWORD(v23) = *v18;
  if ( (_BYTE)v23 != 58 )
  {
    if ( (_BYTE)v23 == 86 )
    {
      v24 = *((_QWORD *)v18 - 12);
      if ( *(_QWORD *)(v24 + 8) == *((_QWORD *)v18 + 1) )
      {
        v25 = (_BYTE *)*((_QWORD *)v18 - 8);
        if ( *v25 <= 0x15u )
        {
          v26 = *((_QWORD *)v18 - 4);
          if ( sub_AD7A80(v25, 1, v20, v21, v22) )
          {
            v28 = *(_QWORD *)(a1 + 8);
            LOBYTE(v24) = v24 == v28;
            LOBYTE(v27) = v26 == v28;
            return v27 | (unsigned int)v24;
          }
        }
      }
    }
    return 0;
  }
  if ( (v18[7] & 0x40) != 0 )
  {
    v29 = (unsigned __int8 *)*((_QWORD *)v18 - 1);
  }
  else
  {
    v23 = 32LL * (*((_DWORD *)v18 + 1) & 0x7FFFFFF);
    v29 = &v18[-v23];
  }
  v30 = *(_QWORD *)(a1 + 8);
  LOBYTE(v3) = *(_QWORD *)v29 == v30;
  LOBYTE(v23) = *((_QWORD *)v29 + 4) == v30;
  v3 |= v23;
  return v3;
}
