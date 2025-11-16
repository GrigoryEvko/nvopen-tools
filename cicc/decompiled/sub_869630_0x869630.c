// Function: sub_869630
// Address: 0x869630
//
__int64 __fastcall sub_869630(__int64 i, int a2)
{
  __int64 v2; // r12
  _QWORD *v3; // rdx
  char v4; // al
  _QWORD *v5; // rcx
  __int64 v6; // rbx
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rsi
  __int64 j; // rax
  __int64 v11; // rax

  v2 = *(_QWORD *)(i + 24);
  if ( a2 )
  {
    v3 = (_QWORD *)i;
    goto LABEL_11;
  }
  v3 = *(_QWORD **)(i + 8);
  if ( v3 )
  {
LABEL_11:
    v5 = v3;
    goto LABEL_12;
  }
  i = *(_QWORD *)i;
  v4 = *(_BYTE *)(i + 16);
  v5 = (_QWORD *)(*(_QWORD *)(qword_4F04C68[0] + 184LL) + 256LL);
  if ( v4 != 54 )
    goto LABEL_13;
LABEL_4:
  if ( *(_QWORD *)(*(_QWORD *)(i + 24) + 16LL) != v2 )
  {
    if ( dword_4F077C4 == 2 )
      goto LABEL_12;
LABEL_6:
    if ( *(char *)(i - 8) >= 0 )
      goto LABEL_12;
    v6 = *(_QWORD *)(i + 24);
    if ( v4 == 6 )
    {
      *v5 = i;
      *(_QWORD *)(i + 8) = v3;
      for ( i = *(_QWORD *)i; ; i = sub_869630(i, 0) )
      {
        for ( ; *(char *)(i - 8) < 0; i = *(_QWORD *)i )
        {
          if ( *(_BYTE *)(i + 16) == 54 && *(_QWORD *)(*(_QWORD *)(i + 24) + 16LL) == v6 )
          {
            *(_BYTE *)(v6 + 143) |= 8u;
            v5 = (_QWORD *)i;
            v3 = (_QWORD *)i;
            goto LABEL_12;
          }
        }
      }
    }
LABEL_8:
    if ( *(_BYTE *)(*(_QWORD *)(v6 + 24) + 140LL) != 12 )
    {
      *v5 = i;
      v5 = (_QWORD *)i;
      *(_QWORD *)(i + 8) = v3;
      v3 = (_QWORD *)i;
      *(_BYTE *)(v6 + 57) |= 1u;
    }
    while ( 1 )
    {
LABEL_12:
      i = *(_QWORD *)i;
      v4 = *(_BYTE *)(i + 16);
      if ( v4 == 54 )
        goto LABEL_4;
LABEL_13:
      if ( v4 == 58 )
      {
        v7 = *(_QWORD *)(i + 24);
LABEL_15:
        if ( *(char *)(v7 - 8) < 0 )
        {
          *v5 = i;
          v5 = (_QWORD *)i;
          *(_QWORD *)(i + 8) = v3;
          v3 = (_QWORD *)i;
        }
      }
      else if ( v4 == 59 )
      {
        v7 = *(_QWORD *)(i + 24);
        if ( (*(_BYTE *)(v7 + 89) & 4) == 0 )
          goto LABEL_15;
        if ( dword_4F077C4 != 2 && *(char *)(i - 8) < 0 )
        {
          v6 = *(_QWORD *)(i + 24);
          goto LABEL_8;
        }
      }
      else
      {
        if ( dword_4F077C4 != 2 )
          goto LABEL_6;
        if ( v4 == 53 )
        {
          v8 = *(_QWORD *)(i + 24);
          if ( (*(_BYTE *)(v8 + 57) & 4) != 0 )
          {
            if ( *(_BYTE *)(v8 + 16) == 11 )
            {
              v9 = *(_QWORD *)(v8 + 24);
              if ( *(_QWORD *)(v9 + 96) == i )
              {
                for ( j = *(_QWORD *)i; j; j = *(_QWORD *)j )
                {
                  if ( *(_BYTE *)(j + 16) == 53 && v9 == *(_QWORD *)(*(_QWORD *)(j + 24) + 24LL) )
                    break;
                }
                *(_QWORD *)(v9 + 96) = j;
              }
            }
          }
          else if ( (*(_BYTE *)(v8 + 57) & 0x20) != 0 && *(char *)(i - 8) < 0 )
          {
            *v5 = i;
            v5 = (_QWORD *)i;
            *(_QWORD *)(i + 8) = v3;
            v3 = (_QWORD *)i;
            *(_BYTE *)(v8 + 57) |= 1u;
          }
        }
      }
    }
  }
  v11 = *(_QWORD *)i;
  *v5 = *(_QWORD *)i;
  if ( v11 )
    *(_QWORD *)(v11 + 8) = v3;
  return *(_QWORD *)i;
}
