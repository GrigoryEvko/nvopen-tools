// Function: sub_1AA6DC0
// Address: 0x1aa6dc0
//
__int64 __fastcall sub_1AA6DC0(__int64 a1, __int64 *a2, __int64 *a3)
{
  __int64 v3; // rax
  __int64 v5; // rdx
  __int64 *v6; // rax
  __int64 v7; // rbx
  __int64 v8; // r12
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // r14
  int v12; // esi
  int v13; // eax
  __int64 v14; // r13
  unsigned __int64 v15; // rax
  __int64 v17; // rbx
  _QWORD *v18; // rax
  _QWORD *v19; // r12
  _QWORD *v20; // rax
  _QWORD *v21; // r14
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  unsigned __int64 v25; // [rsp+0h] [rbp-40h]

  v3 = *(_QWORD *)(a1 + 48);
  if ( !v3 )
    BUG();
  if ( *(_BYTE *)(v3 - 8) != 77 )
  {
    v17 = *(_QWORD *)(a1 + 8);
    while ( v17 )
    {
      v18 = sub_1648700(v17);
      v17 = *(_QWORD *)(v17 + 8);
      v19 = v18;
      if ( (unsigned __int8)(*((_BYTE *)v18 + 16) - 25) <= 9u )
      {
        while ( v17 )
        {
          v20 = sub_1648700(v17);
          v17 = *(_QWORD *)(v17 + 8);
          v21 = v20;
          if ( (unsigned __int8)(*((_BYTE *)v20 + 16) - 25) <= 9u )
          {
            if ( !v17 )
            {
LABEL_26:
              v7 = v19[5];
              v8 = v21[5];
              goto LABEL_7;
            }
            while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v17) + 16) - 25) > 9u )
            {
              v17 = *(_QWORD *)(v17 + 8);
              if ( !v17 )
                goto LABEL_26;
            }
            return 0;
          }
        }
        return 0;
      }
    }
    return 0;
  }
  if ( (*(_DWORD *)(v3 - 4) & 0xFFFFFFF) != 2 )
    return 0;
  v5 = v3 - 72;
  if ( (*(_BYTE *)(v3 - 1) & 0x40) != 0 )
    v5 = *(_QWORD *)(v3 - 32);
  v6 = (__int64 *)(v5 + 24LL * *(unsigned int *)(v3 + 32) + 8);
  v7 = *v6;
  v8 = v6[1];
LABEL_7:
  v9 = sub_157EBA0(v7);
  if ( *(_BYTE *)(v9 + 16) != 26 )
    return 0;
  v25 = v9;
  v10 = sub_157EBA0(v8);
  v11 = v10;
  if ( *(_BYTE *)(v10 + 16) != 26 )
    return 0;
  v12 = *(_DWORD *)(v10 + 20) & 0xFFFFFFF;
  v13 = *(_DWORD *)(v25 + 20) & 0xFFFFFFF;
  if ( v12 == 3 )
  {
    if ( v13 == 3 )
      return 0;
    goto LABEL_28;
  }
  if ( v13 == 3 )
  {
    v24 = v8;
    v11 = v25;
    v8 = v7;
    v7 = v24;
LABEL_28:
    if ( sub_157F0B0(v7) )
    {
      v22 = *(_QWORD *)(v11 - 24);
      if ( v22 && a1 == v22 && v7 == *(_QWORD *)(v11 - 48) )
      {
        *a2 = v8;
        *a3 = v7;
      }
      else
      {
        if ( v7 != v22 )
          return 0;
        v23 = *(_QWORD *)(v11 - 48);
        if ( !v23 || a1 != v23 )
          return 0;
        *a2 = v7;
        *a3 = v8;
      }
      return *(_QWORD *)(v11 - 72);
    }
    return 0;
  }
  v14 = sub_157F0B0(v7);
  if ( !v14 )
    return 0;
  if ( v14 != sub_157F0B0(v8) )
    return 0;
  v15 = sub_157EBA0(v14);
  if ( *(_BYTE *)(v15 + 16) != 26 )
    return 0;
  if ( *(_QWORD *)(v15 - 24) == v7 )
  {
    *a2 = v7;
    *a3 = v8;
  }
  else
  {
    *a2 = v8;
    *a3 = v7;
  }
  return *(_QWORD *)(v15 - 72);
}
