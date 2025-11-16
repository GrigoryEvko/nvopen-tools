// Function: sub_713B40
// Address: 0x713b40
//
__int64 __fastcall sub_713B40(__int64 a1, __int64 a2, _DWORD *a3, __int64 a4)
{
  _BOOL4 v6; // r8d
  __int64 *v7; // rcx
  __int64 *v8; // rax
  unsigned __int8 v9; // si
  unsigned __int8 v10; // dl
  __int64 v11; // rsi
  __int64 v12; // rdx
  __int64 v13; // rsi
  __int64 v15; // rdx
  __int64 v16; // rsi
  __int64 v17; // rdx
  __int64 v18; // rax
  _DWORD v19[9]; // [rsp+Ch] [rbp-24h] BYREF

  v6 = sub_70E2C0(a1, a2, v19, a4);
  if ( !v6 )
    return v6;
  if ( *(_BYTE *)(a1 + 173) != 1 )
  {
    v7 = *(__int64 **)(a1 + 200);
    v8 = *(__int64 **)(a2 + 200);
    if ( v7 && v8 )
    {
      v6 = 0;
      do
      {
        v10 = *((_BYTE *)v7 + 8);
        if ( (v10 & 2) != 0 || (v9 = *((_BYTE *)v8 + 8), (v9 & 2) != 0) )
        {
          if ( v6 )
            return 0;
          if ( ((*((_BYTE *)v8 + 8) ^ v10) & 2) != 0 )
            return v6;
          v11 = v7[2];
          v12 = v8[2];
          if ( v11 != v12 )
          {
            v13 = *(_QWORD *)(v11 + 64);
            if ( *(_QWORD *)(v12 + 64) != v13 || !v13 )
              return v6;
          }
        }
        else
        {
          if ( ((v9 | v10) & 1) != 0 )
          {
            if ( ((v9 ^ v10) & 1) == 0 )
            {
              if ( v7[2] != v8[2] )
                v6 = 1;
              goto LABEL_11;
            }
          }
          else
          {
            v15 = v7[2];
            v16 = v8[2];
            if ( v15 == v16 )
              goto LABEL_11;
            if ( v15 )
            {
              if ( v16 )
              {
                if ( dword_4F07588 )
                {
                  v17 = *(_QWORD *)(v15 + 32);
                  if ( *(_QWORD *)(v16 + 32) == v17 )
                  {
                    if ( v17 )
                      goto LABEL_11;
                  }
                }
              }
            }
          }
          v6 = 1;
        }
LABEL_11:
        v7 = (__int64 *)*v7;
        v8 = (__int64 *)*v8;
      }
      while ( v7 && v8 );
    }
    v18 = *(_QWORD *)(a2 + 192);
    if ( *(_QWORD *)(a1 + 192) == v18 )
    {
      *a3 = 0;
      return 1;
    }
    else
    {
      if ( *(_QWORD *)(a1 + 192) >= v18 )
        *a3 = 1;
      else
        *a3 = -1;
      return 1;
    }
  }
  *a3 = sub_621060(a1, a2);
  return 1;
}
