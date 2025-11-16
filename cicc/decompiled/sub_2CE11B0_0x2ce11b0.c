// Function: sub_2CE11B0
// Address: 0x2ce11b0
//
void __fastcall sub_2CE11B0(_QWORD *a1, char *a2)
{
  _BYTE *v3; // rsi
  char v4; // dl
  _BYTE *v5; // rsi
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // rax
  unsigned int v9; // r12d
  __int64 v10; // rax
  __int64 v11; // rax
  _BYTE *v12; // rsi
  _BYTE *v13; // rsi
  _QWORD v14[2]; // [rsp+8h] [rbp-38h] BYREF
  unsigned int v15; // [rsp+1Ch] [rbp-24h] BYREF
  char *v16; // [rsp+20h] [rbp-20h] BYREF
  _QWORD v17[3]; // [rsp+28h] [rbp-18h] BYREF

  v14[0] = a2;
  if ( *a2 == 82 )
  {
    v16 = a2;
    if ( *(_BYTE *)(*(_QWORD *)(*((_QWORD *)a2 - 8) + 8LL) + 8LL) == 14
      && *(_BYTE *)(*(_QWORD *)(*((_QWORD *)a2 - 4) + 8LL) + 8LL) == 14 )
    {
      v3 = (_BYTE *)a1[10];
      if ( v3 == (_BYTE *)a1[11] )
      {
        sub_2CE1020((__int64)(a1 + 9), v3, &v16);
      }
      else
      {
        if ( v3 )
        {
          *(_QWORD *)v3 = a2;
          v3 = (_BYTE *)a1[10];
        }
        a1[10] = v3 + 8;
      }
    }
    return;
  }
  v16 = 0;
  v4 = *a2;
  if ( *a2 == 61 || v4 == 62 )
  {
    if ( *(_DWORD *)(*(_QWORD *)(*((_QWORD *)a2 - 4) + 8LL) + 8LL) >> 8 )
      return;
  }
  else
  {
    if ( v4 == 85 )
    {
      v6 = *((_QWORD *)a2 - 4);
      if ( v6 )
      {
        if ( !*(_BYTE *)v6 )
        {
          v7 = *((_QWORD *)a2 + 10);
          if ( *(_QWORD *)(v6 + 24) == v7 && (*(_BYTE *)(v6 + 33) & 0x20) != 0 )
          {
            v17[0] = a2;
            v8 = *((_QWORD *)a2 - 4);
            if ( !v8 || *(_BYTE *)v8 || v7 != *(_QWORD *)(v8 + 24) )
              BUG();
            v9 = *(_DWORD *)(v8 + 36);
            v15 = 0;
            if ( (unsigned __int8)sub_2CE0320((__int64)a1, v9, &v15) )
            {
              v10 = *(_QWORD *)(*(_QWORD *)(v17[0]
                                          + 32 * (v15 - (unsigned __int64)(*(_DWORD *)(v17[0] + 4LL) & 0x7FFFFFF)))
                              + 8LL);
              if ( *(_BYTE *)(v10 + 8) == 14 && !(*(_DWORD *)(v10 + 8) >> 8) )
              {
                v13 = (_BYTE *)a1[4];
                if ( v13 == (_BYTE *)a1[5] )
                {
                  sub_24454E0((__int64)(a1 + 3), v13, v14);
                }
                else
                {
                  if ( v13 )
                    *(_QWORD *)v13 = v14[0];
                  a1[4] += 8LL;
                }
              }
              if ( v9 == 241 || v9 == 238 )
              {
                v11 = *(_QWORD *)(*(_QWORD *)(v17[0] + 32 * (1LL - (*(_DWORD *)(v17[0] + 4LL) & 0x7FFFFFF))) + 8LL);
                if ( *(_BYTE *)(v11 + 8) == 14 && !(*(_DWORD *)(v11 + 8) >> 8) )
                {
                  v12 = (_BYTE *)a1[7];
                  if ( v12 == (_BYTE *)a1[8] )
                  {
                    sub_2CB4B10((__int64)(a1 + 6), v12, v17);
                  }
                  else
                  {
                    if ( v12 )
                      *(_QWORD *)v12 = v17[0];
                    a1[7] += 8LL;
                  }
                }
              }
            }
            return;
          }
        }
      }
    }
    v17[0] = 0;
    if ( (unsigned __int8)(*a2 - 65) > 1u )
      return;
  }
  v5 = (_BYTE *)a1[4];
  if ( v5 == (_BYTE *)a1[5] )
  {
    sub_24454E0((__int64)(a1 + 3), v5, v14);
  }
  else
  {
    if ( v5 )
    {
      *(_QWORD *)v5 = a2;
      v5 = (_BYTE *)a1[4];
    }
    a1[4] = v5 + 8;
  }
}
