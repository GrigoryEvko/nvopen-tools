// Function: sub_18DCB50
// Address: 0x18dcb50
//
char __fastcall sub_18DCB50(int a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 v6; // r12
  unsigned __int8 v7; // al
  unsigned int v8; // edi
  char result; // al
  int v10; // edi
  int v11; // eax
  unsigned int v12; // eax
  __int64 v13; // rdi
  int v14; // edi
  unsigned __int8 v15; // al
  __int64 v16; // rdi
  int v17; // eax
  int v18; // edi
  unsigned __int8 v19; // al

  if ( a2 == a3 )
    return 1;
  v6 = a2;
  switch ( a1 )
  {
    case 0:
      v11 = sub_14399D0(a2);
      if ( v11 > 8 )
      {
        if ( v11 == 24 )
          return 0;
      }
      else if ( v11 > 6 )
      {
        return 0;
      }
      return sub_18DC540(a2, a3, a4, v11);
    case 1:
      return (unsigned int)sub_14399D0(a2) - 7 <= 1;
    case 2:
      v12 = sub_14399D0(a2);
      if ( v12 == 8 )
        goto LABEL_27;
      if ( v12 == 24 || v12 == 7 )
        return 0;
      return sub_18DC2F0(a2, a3, a4, v12);
    case 3:
      result = 0;
      if ( *(_BYTE *)(a2 + 16) != 78 )
        return result;
      v16 = *(_QWORD *)(a2 - 24);
      if ( *(_BYTE *)(v16 + 16) )
        return result;
      v17 = sub_1438F00(v16);
      if ( v17 <= 1 )
      {
        if ( v17 >= 0 )
        {
          do
          {
            v18 = 23;
            v6 = sub_1649C60(*(_QWORD *)(v6 - 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF)));
            v19 = *(_BYTE *)(v6 + 16);
            if ( v19 > 0x17u )
            {
              if ( v19 == 78 )
              {
                v18 = 21;
                if ( !*(_BYTE *)(*(_QWORD *)(v6 - 24) + 16LL) )
                  v18 = sub_1438F00(*(_QWORD *)(v6 - 24));
              }
              else
              {
                v18 = 2 * (v19 != 29) + 21;
              }
            }
          }
          while ( (unsigned __int8)sub_1439C90(v18) );
          return a3 == v6;
        }
        return 0;
      }
      if ( (unsigned int)(v17 - 7) > 1 )
        return 0;
LABEL_27:
      result = 1;
      break;
    case 4:
      v7 = *(_BYTE *)(a2 + 16);
      if ( v7 <= 0x17u )
        return sub_1439D20(23);
      if ( v7 != 78 )
        goto LABEL_5;
      v13 = *(_QWORD *)(a2 - 24);
      if ( *(_BYTE *)(v13 + 16) )
        return sub_1439D20(21);
      v8 = sub_1438F00(v13);
      if ( v8 > 1 )
        return sub_1439D20(v8);
      do
      {
        v14 = 23;
        v6 = sub_1649C60(*(_QWORD *)(v6 - 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF)));
        v15 = *(_BYTE *)(v6 + 16);
        if ( v15 > 0x17u )
        {
          if ( v15 == 78 )
          {
            v14 = 21;
            if ( !*(_BYTE *)(*(_QWORD *)(v6 - 24) + 16LL) )
              v14 = sub_1438F00(*(_QWORD *)(v6 - 24));
          }
          else
          {
            v14 = 2 * (v15 != 29) + 21;
          }
        }
      }
      while ( (unsigned __int8)sub_1439C90(v14) );
      return a3 == v6;
    case 5:
      v7 = *(_BYTE *)(a2 + 16);
      v8 = 23;
      if ( v7 > 0x17u )
      {
        if ( v7 == 78 )
        {
          v8 = 21;
          if ( !*(_BYTE *)(*(_QWORD *)(a2 - 24) + 16LL) )
          {
            v10 = sub_1438F00(*(_QWORD *)(a2 - 24));
            return sub_1439D20(v10);
          }
        }
        else
        {
LABEL_5:
          v8 = 2 * (v7 != 29) + 21;
        }
      }
      return sub_1439D20(v8);
  }
  return result;
}
