// Function: sub_19C1150
// Address: 0x19c1150
//
__int64 __fastcall sub_19C1150(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rax
  __int64 v8; // rdx
  unsigned int v9; // eax
  __int64 v10; // r13
  __int64 v11; // rcx
  __int64 v12; // rdx
  __int64 v14; // rax
  int v15; // esi
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rdx

  v5 = a4 - a2;
  v8 = (a4 - a2) >> 2;
  if ( v8 > 0 )
  {
    v9 = 2 * a2 + 8;
    v10 = a2 + 4 * v8;
    while ( 1 )
    {
      v12 = v9 - 6;
      if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
      {
        v11 = *(_QWORD *)(a1 - 8);
        if ( *(_QWORD *)(v11 + 24 * v12) == a5 )
          return a1;
      }
      else
      {
        v11 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
        if ( *(_QWORD *)(v11 + 24 * v12) == a5 )
          return a1;
      }
      if ( *(_QWORD *)(v11 + 24LL * (v9 - 4)) == a5 )
        return a1;
      if ( *(_QWORD *)(v11 + 24LL * (v9 - 2)) == a5 )
        return a1;
      a2 += 4;
      if ( *(_QWORD *)(v11 + 24LL * v9) == a5 )
        return a1;
      v9 += 8;
      if ( v10 == a2 )
      {
        v5 = a4 - a2;
        break;
      }
    }
  }
  switch ( v5 )
  {
    case 2LL:
      v15 = a2 + 1;
      v16 = (unsigned int)(2 * v15);
      if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
      {
LABEL_19:
        if ( a5 == *(_QWORD *)(*(_QWORD *)(a1 - 8) + 24 * v16) )
          return a1;
        v17 = (unsigned int)(2 * v15 + 2);
        goto LABEL_21;
      }
LABEL_27:
      if ( *(_QWORD *)(a1 + 24 * (v16 - (*(_DWORD *)(a1 + 20) & 0xFFFFFFF))) == a5 )
        return a1;
      v17 = (unsigned int)(2 * v15 + 2);
      goto LABEL_23;
    case 3LL:
      v14 = (unsigned int)(2 * (a2 + 1));
      if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
      {
        if ( *(_QWORD *)(*(_QWORD *)(a1 - 8) + 24 * v14) == a5 )
          return a1;
        v15 = a2 + 2;
        v16 = (unsigned int)(2 * v15);
        goto LABEL_19;
      }
      if ( *(_QWORD *)(a1 + 24 * (v14 - (*(_DWORD *)(a1 + 20) & 0xFFFFFFF))) == a5 )
        return a1;
      v15 = a2 + 2;
      v16 = (unsigned int)(2 * v15);
      goto LABEL_27;
    case 1LL:
      v17 = (unsigned int)(2 * a2 + 2);
      if ( (*(_BYTE *)(a1 + 23) & 0x40) == 0 )
      {
LABEL_23:
        v18 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
LABEL_24:
        if ( *(_QWORD *)(v18 + 24 * v17) != a5 )
          return a3;
        return a1;
      }
LABEL_21:
      v18 = *(_QWORD *)(a1 - 8);
      goto LABEL_24;
  }
  return a3;
}
