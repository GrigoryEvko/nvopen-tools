// Function: sub_7A6F40
// Address: 0x7a6f40
//
__int64 __fastcall sub_7A6F40(__int64 a1, __int64 a2, unsigned int a3, unsigned int a4, __int64 a5)
{
  __int64 i; // rbx
  __int64 v7; // r15
  __int64 k; // rdi
  __int64 v9; // rax
  __int64 v10; // r14
  __int64 v11; // rcx
  __int64 *j; // r12
  char v13; // dl
  __int64 v14; // rcx
  _QWORD *v15; // rax
  __int64 v16; // rdi

  if ( (unsigned __int8)(*(_BYTE *)(a1 + 140) - 9) > 2u
    || *(_QWORD *)a1 && (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)a1 + 96LL) + 180LL) & 0x40) == 0 )
  {
    return 0;
  }
  for ( i = *(_QWORD *)(a2 + 40); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v7 = *(_QWORD *)(a1 + 160);
  if ( !v7 )
  {
LABEL_17:
    for ( j = **(__int64 ***)(a1 + 168); j; j = (__int64 *)*j )
    {
      if ( j[13] )
        continue;
      v13 = *((_BYTE *)j + 96);
      if ( (v13 & 0x40) == 0 )
        continue;
      v14 = a4;
      if ( !a4 )
      {
        if ( (v13 & 2) != 0 )
          continue;
        v15 = (_QWORD *)j[14];
        if ( !v15 )
          continue;
        while ( 1 )
        {
          v14 = *(_QWORD *)(v15[1] + 16LL);
          if ( (*(_BYTE *)(v14 + 96) & 2) == 0 )
            break;
          v15 = (_QWORD *)*v15;
          if ( !v15 )
            goto LABEL_31;
        }
      }
      if ( !a3 )
      {
        v16 = j[5];
        if ( v16 == i || (unsigned int)sub_8D97D0(v16, i, 0, v14, a5) )
          return 1;
        v13 = *((_BYTE *)j + 96);
      }
      if ( (v13 & 1) != 0 && (unsigned int)sub_7A6F40(j[5], a2, a3, a4) )
        return 1;
LABEL_31:
      ;
    }
    return 0;
  }
  while ( 1 )
  {
    for ( k = *(_QWORD *)(v7 + 120); *(_BYTE *)(k + 140) == 12; k = *(_QWORD *)(k + 160) )
      ;
    v9 = sub_7A6790(k);
    v10 = v9;
    if ( (*(_BYTE *)(v7 + 144) & 0x40) == 0
      && !(unsigned int)sub_8D3410(v9)
      && *(_QWORD *)(v7 + 128) <= 0xFu
      && (unsigned __int8)(*(_BYTE *)(v10 + 140) - 9) <= 2u
      && (v10 == i || (unsigned int)sub_8D97D0(v10, i, 0, v11, a5) || (unsigned int)sub_7A6F40(v10, a2, 0, 1)) )
    {
      return 1;
    }
    v7 = *(_QWORD *)(v7 + 112);
    if ( !v7 )
      goto LABEL_17;
  }
}
