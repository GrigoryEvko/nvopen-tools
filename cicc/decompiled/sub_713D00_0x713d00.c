// Function: sub_713D00
// Address: 0x713d00
//
__int64 __fastcall sub_713D00(__int64 a1, __int64 a2, _DWORD *a3)
{
  __int64 v4; // r14
  __int64 v5; // rax
  unsigned int v6; // r8d
  int v8; // eax
  __int64 *v9; // rax
  __int64 *v10; // rcx
  unsigned __int8 v11; // si
  unsigned __int8 v12; // dl
  __int64 v13; // rsi
  __int64 v14; // rdx
  __int64 v15; // rsi
  __int64 v16; // rdx
  __int64 v17; // rsi
  __int64 v18; // rdx
  _DWORD v19[13]; // [rsp+Ch] [rbp-34h] BYREF

  v19[0] = 0;
  v4 = sub_70CC20(a1, v19);
  v5 = sub_70CC20(a2, v19);
  v6 = 0;
  if ( !v19[0] )
  {
    if ( *(_BYTE *)(a1 + 173) == 1 )
    {
      v8 = sub_621060(a1, a2);
      v6 = 1;
      *a3 = v8 == 0;
    }
    else if ( v4 == v5 && *(_QWORD *)(a1 + 192) == *(_QWORD *)(a2 + 192) )
    {
      v9 = *(__int64 **)(a2 + 200);
      v10 = *(__int64 **)(a1 + 200);
      if ( v9 && v10 )
      {
        do
        {
          v12 = *((_BYTE *)v10 + 8);
          if ( (v12 & 2) != 0 || (v11 = *((_BYTE *)v9 + 8), (v11 & 2) != 0) )
          {
            if ( v6 )
              return 0;
            if ( ((*((_BYTE *)v9 + 8) ^ v12) & 2) != 0 )
              return v6;
            v13 = v10[2];
            v14 = v9[2];
            if ( v13 != v14 )
            {
              v15 = *(_QWORD *)(v13 + 64);
              if ( *(_QWORD *)(v14 + 64) != v15 || !v15 )
                return v6;
            }
          }
          else
          {
            if ( ((v11 | v12) & 1) != 0 )
            {
              if ( ((v11 ^ v12) & 1) == 0 )
              {
                if ( v10[2] != v9[2] )
                  v6 = 1;
                goto LABEL_16;
              }
            }
            else
            {
              v16 = v10[2];
              v17 = v9[2];
              if ( v16 == v17 )
                goto LABEL_16;
              if ( v16 )
              {
                if ( v17 )
                {
                  if ( dword_4F07588 )
                  {
                    v18 = *(_QWORD *)(v16 + 32);
                    if ( *(_QWORD *)(v17 + 32) == v18 )
                    {
                      if ( v18 )
                        goto LABEL_16;
                    }
                  }
                }
              }
            }
            v6 = 1;
          }
LABEL_16:
          v10 = (__int64 *)*v10;
          v9 = (__int64 *)*v9;
        }
        while ( v10 && v9 );
      }
      *a3 = 1;
      return 1;
    }
    else
    {
      *a3 = 0;
      return 1;
    }
  }
  return v6;
}
