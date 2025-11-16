// Function: sub_631DE0
// Address: 0x631de0
//
__int64 __fastcall sub_631DE0(__int64 *a1, __int64 a2, _DWORD *a3)
{
  __int64 v5; // rdi
  unsigned int v6; // r15d
  __int64 v7; // kr00_8
  __int64 v8; // rdi
  __int64 j; // r14
  __int64 v11; // rbx
  unsigned __int64 v12; // r14
  __int64 i; // rax
  unsigned __int64 v14; // [rsp+0h] [rbp-50h]
  unsigned int v15; // [rsp+Ch] [rbp-44h]
  unsigned __int64 v16; // [rsp+10h] [rbp-40h]
  unsigned __int64 v17; // [rsp+18h] [rbp-38h]

  v5 = *a1;
  v15 = *(_BYTE *)(a2 + 168) & 7;
  v17 = qword_4F06B40[*(_BYTE *)(a2 + 168) & 7];
  v6 = sub_8DBE70(v5);
  if ( a3 )
    *a3 = 0;
  if ( !v6 )
  {
    v7 = v5;
    v8 = *a1;
    switch ( *(_BYTE *)(a2 + 168) & 7 )
    {
      case 0:
        if ( !(unsigned int)sub_8D3440(v8) )
          return v6;
        break;
      case 1:
        if ( !(unsigned int)sub_8D34E0(v8) )
          return v6;
        break;
      case 2:
        if ( (unsigned int)sub_8D3590(v8) )
          break;
        if ( (unsigned int)sub_8D3440(*a1) )
        {
          for ( i = sub_8D4050(*a1); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
            ;
          if ( *(_BYTE *)(i + 160) != 1 )
            break;
        }
        return v6;
      case 3:
        if ( !(unsigned int)sub_8D3610(v8) )
          return v6;
        break;
      case 4:
        if ( !(unsigned int)sub_8D36C0(v8) )
          return v6;
        break;
      default:
        sub_721090(v7);
    }
  }
  for ( j = *a1; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
    ;
  v14 = *(_QWORD *)(a2 + 176);
  v16 = v14 / v17;
  if ( (unsigned int)sub_8D23B0(j) )
  {
    while ( *(_BYTE *)(j + 140) == 12 )
      j = *(_QWORD *)(j + 160);
    v11 = sub_7259C0(8);
    sub_73C230(j, v11);
    *(_QWORD *)(v11 + 176) = v16;
    if ( v17 > v14 )
      *(_BYTE *)(v11 + 169) |= 0x20u;
    v6 = 1;
    sub_8D6090(v11);
    *a1 = v11;
    return v6;
  }
  if ( (*(_WORD *)(j + 168) & 0x180) != 0 )
    return 1;
  if ( v6 )
  {
    if ( dword_4F077BC )
    {
      if ( !(_DWORD)qword_4F077B4 )
      {
        if ( qword_4F077A8 )
          return 1;
        goto LABEL_21;
      }
      goto LABEL_32;
    }
    if ( (_DWORD)qword_4F077B4 )
    {
LABEL_32:
      if ( qword_4F077A0 )
        return 1;
    }
  }
LABEL_21:
  v12 = *(_QWORD *)(j + 176);
  *(_QWORD *)(a2 + 128) = sub_73C8D0(v15, v12);
  if ( v16 < v12 )
  {
    *(_BYTE *)(a2 + 170) |= 0x60u;
    return 1;
  }
  if ( v16 <= v12 )
    return 1;
  if ( dword_4F077C4 == 2 || (v6 = 1, v16 - 1 != v12) )
  {
    v6 = 0;
    if ( a3 )
    {
      *a3 = 1;
      v6 = 1;
    }
  }
  *(_QWORD *)(a2 + 176) = v12 * v17;
  return v6;
}
