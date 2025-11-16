// Function: sub_11A2E80
// Address: 0x11a2e80
//
__int64 __fastcall sub_11A2E80(_QWORD **a1, int a2, unsigned __int8 *a3)
{
  __int64 v4; // rcx
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rcx
  __int64 v8; // rcx
  __int64 v9; // rcx
  __int64 v10; // rcx
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx

  if ( a2 + 29 != *a3 )
    return 0;
  v4 = *((_QWORD *)a3 - 8);
  v5 = *(_QWORD *)(v4 + 16);
  if ( !v5 || *(_QWORD *)(v5 + 8) || *(_BYTE *)v4 != 68 || (v8 = *(_QWORD *)(v4 - 32)) == 0 )
  {
    v6 = *((_QWORD *)a3 - 4);
    v7 = *(_QWORD *)(v6 + 16);
    if ( !v7 || *(_QWORD *)(v7 + 8) )
      return 0;
    goto LABEL_13;
  }
  **a1 = v8;
  v6 = *((_QWORD *)a3 - 4);
  v9 = *(_QWORD *)(v6 + 16);
  if ( !v9 || *(_QWORD *)(v9 + 8) )
    return 0;
  if ( *(_BYTE *)v6 != 69 || (v14 = *(_QWORD *)(v6 - 32)) == 0 )
  {
LABEL_13:
    if ( *(_BYTE *)v6 == 68 )
    {
      v10 = *(_QWORD *)(v6 - 32);
      if ( v10 )
      {
        **a1 = v10;
        v11 = *((_QWORD *)a3 - 8);
        v12 = *(_QWORD *)(v11 + 16);
        if ( v12 )
        {
          if ( !*(_QWORD *)(v12 + 8) && *(_BYTE *)v11 == 69 )
          {
            v13 = *(_QWORD *)(v11 - 32);
            if ( v13 )
            {
              *a1[1] = v13;
              return 1;
            }
          }
        }
      }
    }
    return 0;
  }
  *a1[1] = v14;
  return 1;
}
