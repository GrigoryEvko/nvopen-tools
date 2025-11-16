// Function: sub_8D1FA0
// Address: 0x8d1fa0
//
__int64 __fastcall sub_8D1FA0(__int64 a1, _DWORD *a2)
{
  char v3; // al
  __int64 *v4; // rax
  char v5; // dl
  unsigned int v6; // r13d
  unsigned __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v10; // rax
  __int64 v11; // rdi
  __int64 *v12[5]; // [rsp+8h] [rbp-28h] BYREF

  v3 = *(_BYTE *)(a1 + 140);
  switch ( v3 )
  {
    case 14:
      if ( *(_BYTE *)(a1 + 160) )
        return 0;
LABEL_22:
      *a2 = 1;
      return 1;
    case 12:
      v7 = *(unsigned __int8 *)(a1 + 184);
      if ( (unsigned __int8)v7 <= 0xCu )
      {
        v8 = 6338;
        if ( _bittest64(&v8, v7) )
          *a2 = 1;
      }
      return 0;
    case 8:
      if ( *(char *)(a1 + 168) >= 0 )
        return 0;
      if ( !(unsigned int)sub_8D19F0(*(_QWORD *)(a1 + 176)) )
        goto LABEL_34;
      goto LABEL_22;
  }
  if ( (unsigned __int8)(v3 - 9) > 2u )
  {
    if ( v3 != 7 )
      return 0;
    if ( !dword_4F06978 )
      return 0;
    v10 = *(_QWORD *)(*(_QWORD *)(a1 + 168) + 56LL);
    if ( !v10 )
      return 0;
    if ( (*(_BYTE *)v10 & 0x61) != 1 )
      return 0;
    v11 = *(_QWORD *)(v10 + 8);
    if ( !v11 )
      return 0;
    if ( !(unsigned int)sub_8D19F0(v11) )
      goto LABEL_34;
    goto LABEL_22;
  }
  if ( (*(_BYTE *)(a1 + 177) & 0x20) == 0 )
    return 0;
  v4 = *(__int64 **)(*(_QWORD *)(a1 + 168) + 168LL);
  v12[0] = v4;
  if ( !v4 )
    return sub_8D1BE0(a1, a2);
  v5 = *((_BYTE *)v4 + 8);
  if ( v5 == 3 )
  {
    sub_72F220(v12);
    v4 = v12[0];
    if ( !v12[0] )
      goto LABEL_34;
    v5 = *((_BYTE *)v12[0] + 8);
  }
  v6 = 0;
LABEL_9:
  if ( v5 == 1 )
    goto LABEL_14;
  while ( 1 )
  {
    v4 = (__int64 *)*v12[0];
    v12[0] = v4;
    if ( !v4 )
      break;
    v5 = *((_BYTE *)v4 + 8);
    if ( v5 != 3 )
      goto LABEL_9;
    sub_72F220(v12);
    v4 = v12[0];
    if ( !v12[0] )
      break;
    if ( *((_BYTE *)v12[0] + 8) == 1 )
    {
LABEL_14:
      if ( (unsigned int)sub_8D19F0(v4[4]) )
      {
        *a2 = 1;
        v6 = 1;
      }
    }
  }
  if ( v6 )
    return v6;
LABEL_34:
  if ( (unsigned __int8)(*(_BYTE *)(a1 + 140) - 9) > 2u || (*(_BYTE *)(a1 + 177) & 0x20) == 0 )
    return 0;
  return sub_8D1BE0(a1, a2);
}
