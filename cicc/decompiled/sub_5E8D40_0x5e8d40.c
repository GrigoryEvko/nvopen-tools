// Function: sub_5E8D40
// Address: 0x5e8d40
//
__int64 __fastcall sub_5E8D40(__int64 *a1, __int64 a2, __int64 a3)
{
  char v4; // al
  __int64 v5; // rax
  __int64 *v7; // rax
  char v8; // dl
  __int64 *v9; // rcx
  _QWORD *v10; // rax

  if ( a1 )
  {
    while ( 1 )
    {
      v4 = *((_BYTE *)a1 + 32);
      if ( (v4 & 0x10) != 0 )
        sub_721090(a1);
      if ( (v4 & 1) != 0 )
      {
        if ( **(_QWORD **)a1[2] == a2 )
          goto LABEL_11;
        goto LABEL_4;
      }
      if ( (v4 & 4) != 0 )
        goto LABEL_12;
      v5 = a1[1];
      if ( v5 )
      {
        if ( (*(_BYTE *)(v5 + 172) & 1) == 0 )
        {
          if ( a2 == **(_QWORD **)v5 )
            goto LABEL_11;
          goto LABEL_4;
        }
LABEL_12:
        if ( !a2 )
          goto LABEL_11;
        a1 = (__int64 *)*a1;
        if ( !a1 )
          return 0;
      }
      else
      {
        v7 = a1;
        while ( 1 )
        {
          v8 = *((_BYTE *)v7 + 33);
          v9 = v7;
          v7 = (__int64 *)v7[2];
          if ( (v8 & 2) == 0 )
            break;
          if ( !v7 )
            goto LABEL_22;
        }
        if ( v7 )
          goto LABEL_19;
LABEL_22:
        v7 = (__int64 *)v9[3];
        if ( !v7 )
          goto LABEL_4;
LABEL_19:
        v10 = (_QWORD *)*v7;
        if ( v10 && a2 == *v10 )
        {
LABEL_11:
          sub_6851C0(1761, a3);
          return 1;
        }
LABEL_4:
        a1 = (__int64 *)*a1;
        if ( !a1 )
          return 0;
      }
    }
  }
  return 0;
}
