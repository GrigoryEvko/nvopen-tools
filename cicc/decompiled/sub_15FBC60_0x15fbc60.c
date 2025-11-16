// Function: sub_15FBC60
// Address: 0x15fbc60
//
bool __fastcall sub_15FBC60(__int64 a1, __int64 a2)
{
  char v2; // dl
  bool result; // al
  char v4; // bl
  __int64 v5; // r12
  char v6; // bl
  int v7; // ebx
  int v8; // eax
  int v9; // r13d
  int v10; // eax
  int v11; // r12d
  int v12; // eax

  v2 = *(_BYTE *)(a1 + 8);
  result = v2 != 0 && v2 != 12;
  if ( result )
  {
    v4 = *(_BYTE *)(a2 + 8);
    result = v4 != 12 && v4 != 0;
    if ( result )
    {
      v5 = a1;
      if ( a1 != a2 )
      {
        if ( v2 == 16 && v4 == 16 )
        {
          if ( *(_QWORD *)(a1 + 32) != *(_QWORD *)(a2 + 32) )
          {
            v7 = sub_1643030(a1);
            v8 = sub_1643030(a2);
            if ( v7 == 0 || v8 == 0 || v8 != v7 )
              return 0;
            goto LABEL_13;
          }
          a2 = *(_QWORD *)(a2 + 24);
          v5 = *(_QWORD *)(a1 + 24);
          v4 = *(_BYTE *)(a2 + 8);
        }
        if ( v4 == 15 )
        {
          v6 = *(_BYTE *)(v5 + 8);
          if ( v6 == 15 )
            return *(_DWORD *)(a2 + 8) >> 8 == *(_DWORD *)(v5 + 8) >> 8;
          v11 = sub_1643030(v5);
          v12 = sub_1643030(a2);
          if ( v11 != 0 && v12 != 0 )
          {
            if ( v11 != v12 )
              return 0;
            return v6 != 9;
          }
          return 0;
        }
        v9 = sub_1643030(v5);
        v10 = sub_1643030(a2);
        result = v9 != v10 || v10 == 0 || v9 == 0;
        if ( result )
          return 0;
        if ( v4 != 9 )
        {
LABEL_13:
          v6 = *(_BYTE *)(v5 + 8);
          return v6 != 9;
        }
      }
    }
  }
  return result;
}
