// Function: sub_2C50020
// Address: 0x2c50020
//
bool __fastcall sub_2C50020(__int64 ***a1, unsigned __int8 **a2)
{
  unsigned __int8 *v2; // r12
  unsigned __int8 v3; // dl
  __int64 v4; // rdi
  __int64 v5; // rcx
  bool result; // al
  __int64 v7; // rbx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // rcx
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // r13
  __int64 v15; // rcx

  if ( !a2 )
    return 1;
  v2 = *a2;
  v3 = **a2;
  v4 = ***a1;
  if ( v3 <= 0x1Cu )
    return *(_BYTE *)v4 == v3;
  v5 = *((_QWORD *)v2 + 2);
  result = 0;
  if ( v5 )
  {
    v7 = *(_QWORD *)(v5 + 8);
    if ( !v7 && *(_BYTE *)v4 == v3 )
    {
      if ( (unsigned __int8)(v3 - 82) > 1u )
      {
        if ( (unsigned int)v3 - 67 > 0xC )
        {
LABEL_19:
          if ( v3 == 86 )
          {
            v15 = *(_QWORD *)(*((_QWORD *)v2 - 12) + 8LL);
            result = 0;
            if ( (unsigned int)*(unsigned __int8 *)(v15 + 8) - 17 <= 1 )
              return *(_QWORD *)(*(_QWORD *)(v4 - 96) + 8LL) == v15;
          }
          else
          {
            result = 1;
            if ( v3 == 85 )
            {
              v10 = *((_QWORD *)v2 - 4);
              result = 0;
              if ( v10 )
              {
                if ( !*(_BYTE *)v10
                  && *(_QWORD *)(v10 + 24) == *((_QWORD *)v2 + 10)
                  && (*(_BYTE *)(v10 + 33) & 0x20) != 0
                  && *(_BYTE *)v4 == 85 )
                {
                  v11 = *(_QWORD *)(v4 - 32);
                  if ( v11 )
                  {
                    if ( !*(_BYTE *)v11 && *(_QWORD *)(v11 + 24) == *(_QWORD *)(v4 + 80) )
                    {
                      if ( (*(_BYTE *)(v11 + 33) & 0x20) != 0 )
                      {
                        if ( *(_DWORD *)(v11 + 36) == *(_DWORD *)(v10 + 36) )
                        {
                          result = 1;
                          if ( (v2[7] & 0x80u) != 0 )
                          {
                            v12 = sub_BD2BC0((__int64)v2);
                            v14 = v12 + v13;
                            if ( (v2[7] & 0x80u) != 0 )
                              v7 = sub_BD2BC0((__int64)v2);
                            return (unsigned int)((v14 - v7) >> 4) == 0;
                          }
                        }
                      }
                      else
                      {
                        return 0;
                      }
                    }
                  }
                }
              }
            }
          }
          return result;
        }
      }
      else
      {
        if ( (*((_WORD *)v2 + 1) & 0x3F) != (*(_WORD *)(v4 + 2) & 0x3F) )
          return result;
        if ( (unsigned int)v3 - 67 > 0xC )
          return 1;
      }
      v8 = *(_QWORD *)(*((_QWORD *)v2 - 4) + 8LL);
      if ( (unsigned int)*(unsigned __int8 *)(v8 + 8) - 17 <= 1 )
        v8 = **(_QWORD **)(v8 + 16);
      v9 = *(_QWORD *)(*(_QWORD *)(v4 - 32) + 8LL);
      if ( (unsigned int)*(unsigned __int8 *)(v9 + 8) - 17 <= 1 )
        v9 = **(_QWORD **)(v9 + 16);
      result = 0;
      if ( v9 == v8 )
        goto LABEL_19;
    }
  }
  return result;
}
