// Function: sub_6236C0
// Address: 0x6236c0
//
__int64 __fastcall sub_6236C0(__int64 *a1, __int64 *a2)
{
  __int64 v3; // rdi
  __int64 result; // rax
  __int64 v5; // r12
  __int64 v6; // rdx
  int i; // r14d
  __int64 v8; // rdx
  __int64 v9; // rdi
  unsigned __int8 v10; // dl
  __int64 j; // rdx

  v3 = *a1;
  result = *a2;
  if ( v3 != *a2 )
  {
    v5 = 0;
    while ( 1 )
    {
      if ( v3 )
      {
        if ( result )
        {
          if ( dword_4F07588 )
          {
            v6 = *(_QWORD *)(v3 + 32);
            if ( *(_QWORD *)(result + 32) == v6 )
            {
              if ( v6 )
                return result;
            }
          }
        }
      }
      if ( *(_BYTE *)(v3 + 140) == 12 )
      {
        for ( i = sub_8D4C10(v3, 1); *(_BYTE *)(v3 + 140) == 12; v3 = *(_QWORD *)(v3 + 160) )
          ;
        result = *a2;
        if ( *a2 == v3 )
          break;
        if ( result )
        {
          if ( dword_4F07588 )
          {
            v8 = *(_QWORD *)(v3 + 32);
            if ( *(_QWORD *)(result + 32) == v8 )
            {
              if ( v8 )
                break;
            }
          }
        }
      }
      v5 = v3;
      v3 = sub_8D48B0(v3, 0);
      result = *a2;
      if ( *a2 == v3 )
        return result;
    }
    if ( (i & 4) != 0 )
    {
      v9 = 644;
      sub_6851C0(644, dword_4F07508);
      if ( i == 4 )
      {
        result = *a2;
      }
      else
      {
        v9 = *a2;
        result = sub_73C570(*a2, i & 0xFFFFFFFB, -1);
        for ( j = result; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
          ;
        *a2 = j;
      }
      if ( !v5 )
      {
        *a1 = result;
        goto LABEL_27;
      }
      v10 = *(_BYTE *)(v5 + 140);
      if ( v10 != 8 )
      {
        if ( v10 > 8u )
        {
          if ( v10 != 13 )
            goto LABEL_30;
          *(_QWORD *)(v5 + 168) = result;
LABEL_27:
          while ( *(_BYTE *)(result + 140) == 12 )
            result = *(_QWORD *)(result + 160);
          *a2 = result;
          return result;
        }
        if ( v10 != 6 && v10 != 7 )
LABEL_30:
          sub_721090(v9);
      }
      *(_QWORD *)(v5 + 160) = result;
      goto LABEL_27;
    }
  }
  return result;
}
