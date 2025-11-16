// Function: sub_1008380
// Address: 0x1008380
//
__int64 __fastcall sub_1008380(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rdi
  unsigned int v4; // eax
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  unsigned int v8; // r12d
  __int64 v9; // r14
  _BYTE *v10; // rdi
  __int64 v11; // rbx
  _QWORD *v12; // rbx
  __int64 v13; // rcx

  if ( *(_BYTE *)a2 <= 0x1Cu )
    return 0;
  v3 = *(_QWORD *)(a2 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v3 + 8) - 17 <= 1 )
    v3 = **(_QWORD **)(v3 + 16);
  LOBYTE(v4) = sub_BCAC40(v3, 1);
  v8 = v4;
  if ( !(_BYTE)v4 )
    return 0;
  if ( *(_BYTE *)a2 != 58 )
  {
    if ( *(_BYTE *)a2 == 86 )
    {
      v9 = *(_QWORD *)(a2 - 96);
      if ( *(_QWORD *)(a2 + 8) == *(_QWORD *)(v9 + 8) )
      {
        v10 = *(_BYTE **)(a2 - 64);
        if ( *v10 <= 0x15u )
        {
          v11 = *(_QWORD *)(a2 - 32);
          if ( sub_AD7A80(v10, 1, v5, v6, v7) )
          {
            if ( v9 == *a1 && v11 == a1[1] )
              return v8;
            if ( v11 == *a1 )
            {
              LOBYTE(v8) = a1[1] == v9;
              return v8;
            }
          }
        }
      }
    }
    return 0;
  }
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v12 = *(_QWORD **)(a2 - 8);
  else
    v12 = (_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v13 = v12[4];
  if ( *v12 != *a1 || v13 != a1[1] )
  {
    if ( v13 == *a1 )
    {
      LOBYTE(v8) = a1[1] == *v12;
      return v8;
    }
    return 0;
  }
  return v8;
}
