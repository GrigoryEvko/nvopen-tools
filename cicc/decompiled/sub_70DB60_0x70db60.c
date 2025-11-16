// Function: sub_70DB60
// Address: 0x70db60
//
__int64 __fastcall sub_70DB60(__int64 a1)
{
  __int64 v1; // r12
  char i; // al
  __int64 v3; // rdx
  unsigned int v4; // r15d
  __int64 v6; // rbx
  char v7; // al
  int v8; // r14d
  _BYTE *v9; // r13
  int v10; // eax

  v1 = sub_8D4130(a1);
  for ( i = *(_BYTE *)(v1 + 140); i == 12; i = *(_BYTE *)(v1 + 140) )
    v1 = *(_QWORD *)(v1 + 160);
  if ( (unsigned __int8)(i - 9) <= 2u )
  {
    v3 = *(_QWORD *)(*(_QWORD *)v1 + 96LL);
    if ( *(_QWORD *)(v3 + 24) && (*(_BYTE *)(v3 + 177) & 2) == 0 )
      return 0;
    v6 = *(_QWORD *)(v3 + 8);
    if ( v6 )
    {
      v7 = *(_BYTE *)(v6 + 80);
      v8 = 0;
      if ( v7 != 17 )
        goto LABEL_10;
      v6 = *(_QWORD *)(v6 + 88);
      if ( v6 )
      {
        v7 = *(_BYTE *)(v6 + 80);
        v8 = 1;
LABEL_10:
        v4 = 0;
        if ( v7 == 20 )
          goto LABEL_15;
LABEL_11:
        v9 = *(_BYTE **)(v6 + 88);
        if ( (v9[206] & 0x10) != 0 )
          goto LABEL_15;
        if ( dword_4F07588 )
        {
          if ( (*(_BYTE *)(v6 + 104) & 1) != 0 )
            v10 = sub_8796F0(v6);
          else
            v10 = (v9[208] & 4) != 0;
          if ( v10 )
          {
LABEL_15:
            while ( v8 )
            {
              v6 = *(_QWORD *)(v6 + 8);
              if ( !v6 )
                break;
              if ( *(_BYTE *)(v6 + 80) != 20 )
                goto LABEL_11;
            }
            return v4;
          }
          if ( (v9[194] & 4) != 0 )
          {
LABEL_14:
            v4 = 1;
            goto LABEL_15;
          }
        }
        else if ( (v9[194] & 4) != 0 )
        {
          goto LABEL_14;
        }
        if ( !(unsigned int)sub_72F500(v9, v1, 0, 1, 1) )
          goto LABEL_15;
        return 0;
      }
    }
    return (*(_BYTE *)(v3 + 177) & 0x40) != 0;
  }
  return sub_8D2530(v1);
}
