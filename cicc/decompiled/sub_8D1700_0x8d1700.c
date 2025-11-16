// Function: sub_8D1700
// Address: 0x8d1700
//
_BOOL8 __fastcall sub_8D1700(__int64 a1, _DWORD *a2, __int64 a3, __int64 a4)
{
  char v4; // al
  __int64 *v5; // rax
  char v6; // dl
  _BOOL8 result; // rax
  char v8; // al
  __int64 v9; // rdi
  __int64 v10; // rax
  __int64 *v11[3]; // [rsp+8h] [rbp-18h] BYREF

  v4 = *(_BYTE *)(a1 + 140);
  if ( v4 == 8 )
  {
    v8 = *(_BYTE *)(a1 + 169);
    if ( (v8 & 1) != 0 && (v8 & 2) == 0 )
    {
      if ( !(unsigned int)sub_73A2D0(*(_QWORD *)(a1 + 176), (_UNKNOWN *__ptr32 *)qword_4F60578, a3, a4) )
        return 0;
LABEL_12:
      result = 1;
      goto LABEL_22;
    }
    if ( *(char *)(a1 + 168) >= 0 )
      return 0;
    v9 = *(_QWORD *)(a1 + 176);
    if ( !v9 )
      return 0;
  }
  else
  {
    if ( (unsigned __int8)(v4 - 9) <= 2u )
    {
      if ( (*(_BYTE *)(a1 + 177) & 0x20) != 0 )
      {
        v5 = *(__int64 **)(*(_QWORD *)(a1 + 168) + 168LL);
        v11[0] = v5;
        if ( v5 )
        {
          v6 = *((_BYTE *)v5 + 8);
          if ( v6 != 3 )
            goto LABEL_6;
          sub_72F220(v11);
          v5 = v11[0];
          if ( v11[0] )
          {
            v6 = *((_BYTE *)v11[0] + 8);
            while ( 1 )
            {
LABEL_6:
              if ( v6 == 1 )
              {
LABEL_11:
                if ( sub_8D1670(v5[4]) )
                  goto LABEL_12;
              }
              while ( 1 )
              {
                v5 = (__int64 *)*v11[0];
                v11[0] = v5;
                if ( !v5 )
                  return 0;
                v6 = *((_BYTE *)v5 + 8);
                if ( v6 != 3 )
                  break;
                sub_72F220(v11);
                v5 = v11[0];
                if ( !v11[0] )
                  return 0;
                if ( *((_BYTE *)v11[0] + 8) == 1 )
                  goto LABEL_11;
              }
            }
          }
        }
      }
      return 0;
    }
    if ( v4 != 7 )
      return 0;
    if ( !dword_4F06978 )
      return 0;
    v10 = *(_QWORD *)(*(_QWORD *)(a1 + 168) + 56LL);
    if ( !v10 )
      return 0;
    if ( (*(_BYTE *)v10 & 0x61) != 1 )
      return 0;
    v9 = *(_QWORD *)(v10 + 8);
    if ( !v9 )
      return 0;
  }
  result = sub_8D1670(v9);
  if ( !result )
    return 0;
LABEL_22:
  *a2 = 1;
  return result;
}
