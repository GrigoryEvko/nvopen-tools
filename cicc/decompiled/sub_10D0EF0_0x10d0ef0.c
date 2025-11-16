// Function: sub_10D0EF0
// Address: 0x10d0ef0
//
__int64 __fastcall sub_10D0EF0(_QWORD **a1, int a2, unsigned __int8 *a3)
{
  __int64 v4; // rcx
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rcx
  __int64 v8; // rcx
  __int64 v9; // rcx
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rcx
  __int64 v16; // rcx
  __int64 v17; // rcx

  if ( a2 + 29 != *a3 )
    return 0;
  v4 = *((_QWORD *)a3 - 8);
  v5 = *(_QWORD *)(v4 + 16);
  if ( !v5 )
    goto LABEL_4;
  if ( *(_QWORD *)(v5 + 8) )
    goto LABEL_4;
  if ( *(_BYTE *)v4 != 57 )
    goto LABEL_4;
  v14 = *(_QWORD *)(v4 - 64);
  if ( !v14 )
    goto LABEL_4;
  **a1 = v14;
  v15 = *(_QWORD *)(v4 - 32);
  if ( !v15 )
    goto LABEL_4;
  *a1[1] = v15;
  v6 = *((_QWORD *)a3 - 4);
  v16 = *(_QWORD *)(v6 + 16);
  if ( !v16 || *(_QWORD *)(v16 + 8) )
    return 0;
  if ( *(_BYTE *)v6 != 58 || (v17 = *(_QWORD *)(v6 - 64)) == 0 )
  {
LABEL_6:
    if ( *(_BYTE *)v6 == 57 )
    {
      v8 = *(_QWORD *)(v6 - 64);
      if ( v8 )
      {
        **a1 = v8;
        v9 = *(_QWORD *)(v6 - 32);
        if ( v9 )
        {
          *a1[1] = v9;
          v10 = *((_QWORD *)a3 - 8);
          v11 = *(_QWORD *)(v10 + 16);
          if ( v11 )
          {
            if ( !*(_QWORD *)(v11 + 8) && *(_BYTE *)v10 == 58 )
            {
              v12 = *(_QWORD *)(v10 - 64);
              if ( v12 )
              {
                *a1[2] = v12;
                v13 = *(_QWORD *)(v10 - 32);
                if ( v13 )
                  goto LABEL_14;
              }
            }
          }
        }
      }
    }
    return 0;
  }
  *a1[2] = v17;
  v13 = *(_QWORD *)(v6 - 32);
  if ( !v13 )
  {
LABEL_4:
    v6 = *((_QWORD *)a3 - 4);
    v7 = *(_QWORD *)(v6 + 16);
    if ( !v7 || *(_QWORD *)(v7 + 8) )
      return 0;
    goto LABEL_6;
  }
LABEL_14:
  *a1[3] = v13;
  return 1;
}
