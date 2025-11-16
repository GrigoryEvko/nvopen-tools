// Function: sub_B4E200
// Address: 0xb4e200
//
__int64 __fastcall sub_B4E200(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rdx
  unsigned int v5; // r12d
  __int64 v7; // r14
  char v8; // al
  __int64 v9; // rdx
  unsigned __int8 v10; // cl
  int v11; // esi
  unsigned __int64 v12; // rax
  __int64 v13; // rax
  unsigned __int8 **v14; // r14
  unsigned __int8 **v15; // rdx
  unsigned __int8 *v16; // r15
  int v17; // eax
  unsigned int v18; // r13d
  unsigned __int64 v19; // rax
  int v20; // eax
  int v21; // r14d
  unsigned int v22; // r15d
  unsigned __int8 **v23; // [rsp+8h] [rbp-38h]

  v4 = *(_QWORD *)(a1 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v4 + 8) - 17 > 1 )
    return 0;
  if ( v4 != *(_QWORD *)(a2 + 8) )
    return 0;
  v7 = *(_QWORD *)(a3 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v7 + 8) - 17 > 1 )
    return 0;
  v5 = sub_BCAC40(*(_QWORD *)(v7 + 24), 32);
  if ( !(_BYTE)v5 )
    return 0;
  v8 = *(_BYTE *)(v7 + 8);
  v9 = *(_QWORD *)(a1 + 8);
  if ( (v8 == 18) != (*(_BYTE *)(v9 + 8) == 18) )
    return 0;
  v10 = *(_BYTE *)a3;
  if ( (unsigned __int8)(*(_BYTE *)a3 - 12) <= 2u )
    return v5;
  if ( v8 == 18 )
    return 0;
  v11 = *(_DWORD *)(v9 + 32);
  if ( v10 == 17 )
  {
    v5 = *(_DWORD *)(a3 + 32);
    if ( v5 <= 0x40 )
    {
      v12 = *(_QWORD *)(a3 + 24);
LABEL_13:
      LOBYTE(v5) = (unsigned int)(2 * v11) > v12;
      return v5;
    }
    v5 -= sub_C444A0(a3 + 24);
    if ( v5 <= 0x40 )
    {
      v12 = **(_QWORD **)(a3 + 24);
      goto LABEL_13;
    }
    return 0;
  }
  if ( v10 != 11 )
  {
    if ( (unsigned int)v10 - 15 <= 1 )
    {
      v21 = *(_DWORD *)(v7 + 32);
      if ( !v21 )
        return v5;
      v22 = 0;
      while ( sub_AC5320(a3, v22) < (unsigned __int64)(unsigned int)(2 * v11) )
      {
        if ( ++v22 == v21 )
          return v5;
      }
    }
    return 0;
  }
  v13 = 4LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF);
  if ( (*(_BYTE *)(a3 + 7) & 0x40) != 0 )
  {
    v14 = *(unsigned __int8 ***)(a3 - 8);
    v15 = &v14[v13];
  }
  else
  {
    v15 = (unsigned __int8 **)a3;
    v14 = (unsigned __int8 **)(a3 - v13 * 8);
  }
  for ( ; v15 != v14; v14 += 4 )
  {
    v16 = *v14;
    v17 = **v14;
    if ( (_BYTE)v17 == 17 )
    {
      v18 = *((_DWORD *)v16 + 8);
      if ( v18 > 0x40 )
      {
        v23 = v15;
        v20 = sub_C444A0(v16 + 24);
        v15 = v23;
        if ( v18 - v20 > 0x40 )
          return 0;
        v19 = **((_QWORD **)v16 + 3);
      }
      else
      {
        v19 = *((_QWORD *)v16 + 3);
      }
      if ( (unsigned int)(2 * v11) <= v19 )
        return 0;
    }
    else if ( (unsigned int)(v17 - 12) > 1 )
    {
      return 0;
    }
  }
  return v5;
}
