// Function: sub_2D698E0
// Address: 0x2d698e0
//
bool __fastcall sub_2D698E0(__int64 a1, int a2, unsigned __int8 *a3)
{
  bool result; // al
  __int64 v4; // rcx
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rcx
  __int64 v8; // rcx
  __int64 v9; // rcx
  __int64 v10; // rcx
  __int64 v11; // rsi
  __int64 v12; // rsi
  __int64 v13; // rcx
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rcx
  __int64 v18; // rcx
  unsigned __int8 *v19; // [rsp-20h] [rbp-20h]

  if ( a2 + 29 != *a3 )
    return 0;
  v4 = *((_QWORD *)a3 - 8);
  v5 = *(_QWORD *)(v4 + 16);
  if ( !v5 || *(_QWORD *)(v5 + 8) || *(_BYTE *)v4 != 68 || (v8 = *(_QWORD *)(v4 - 32)) == 0 )
  {
LABEL_4:
    v6 = *((_QWORD *)a3 - 4);
    v7 = *(_QWORD *)(v6 + 16);
    if ( v7 && !*(_QWORD *)(v7 + 8) )
      goto LABEL_20;
    return 0;
  }
  **(_QWORD **)a1 = v8;
  v6 = *((_QWORD *)a3 - 4);
  v9 = *(_QWORD *)(v6 + 16);
  if ( !v9 || *(_QWORD *)(v9 + 8) )
    return 0;
  if ( *(_BYTE *)v6 != 54
    || (v10 = *(_QWORD *)(v6 - 64), (v11 = *(_QWORD *)(v10 + 16)) == 0)
    || *(_QWORD *)(v11 + 8)
    || *(_BYTE *)v10 != 68
    || (v12 = *(_QWORD *)(v10 - 32)) == 0 )
  {
LABEL_20:
    if ( *(_BYTE *)v6 == 68 )
    {
      v13 = *(_QWORD *)(v6 - 32);
      if ( v13 )
      {
        **(_QWORD **)a1 = v13;
        v14 = *((_QWORD *)a3 - 8);
        v15 = *(_QWORD *)(v14 + 16);
        if ( v15 )
        {
          if ( !*(_QWORD *)(v15 + 8) && *(_BYTE *)v14 == 54 )
          {
            v16 = *(_QWORD *)(v14 - 64);
            v17 = *(_QWORD *)(v16 + 16);
            if ( v17 )
            {
              if ( !*(_QWORD *)(v17 + 8) && *(_BYTE *)v16 == 68 )
              {
                v18 = *(_QWORD *)(v16 - 32);
                if ( v18 )
                {
                  **(_QWORD **)(a1 + 8) = v18;
                  return sub_F17ED0((_QWORD *)(a1 + 16), *(_QWORD *)(v14 - 32));
                }
              }
            }
          }
        }
      }
    }
    return 0;
  }
  v19 = a3;
  **(_QWORD **)(a1 + 8) = v12;
  result = sub_F17ED0((_QWORD *)(a1 + 16), *(_QWORD *)(v6 - 32));
  if ( !result )
  {
    a3 = v19;
    goto LABEL_4;
  }
  return result;
}
