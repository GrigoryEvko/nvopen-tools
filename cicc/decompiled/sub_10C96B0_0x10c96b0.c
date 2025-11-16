// Function: sub_10C96B0
// Address: 0x10c96b0
//
__int64 __fastcall sub_10C96B0(_QWORD **a1, int a2, unsigned __int8 *a3)
{
  __int64 v4; // rcx
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rcx
  __int64 v8; // rcx
  __int64 v9; // rcx

  if ( a2 + 29 != *a3 )
    return 0;
  v4 = *((_QWORD *)a3 - 8);
  v5 = *(_QWORD *)(v4 + 16);
  if ( v5 && !*(_QWORD *)(v5 + 8) && *(_BYTE *)v4 == 68 && (v9 = *(_QWORD *)(v4 - 32)) != 0 )
  {
    **a1 = v9;
    v6 = *((_QWORD *)a3 - 4);
    if ( v6 )
      goto LABEL_15;
  }
  else
  {
    v6 = *((_QWORD *)a3 - 4);
  }
  v7 = *(_QWORD *)(v6 + 16);
  if ( !v7 )
    return 0;
  if ( *(_QWORD *)(v7 + 8) )
    return 0;
  if ( *(_BYTE *)v6 != 68 )
    return 0;
  v8 = *(_QWORD *)(v6 - 32);
  if ( !v8 )
    return 0;
  **a1 = v8;
  v6 = *((_QWORD *)a3 - 8);
  if ( !v6 )
    return 0;
LABEL_15:
  *a1[1] = v6;
  return 1;
}
