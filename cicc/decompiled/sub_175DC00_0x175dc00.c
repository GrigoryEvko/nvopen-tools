// Function: sub_175DC00
// Address: 0x175dc00
//
char __fastcall sub_175DC00(_QWORD **a1, __int64 a2, __int64 a3, __int64 a4)
{
  char v4; // al
  __int64 v6; // rax
  __int64 v7; // r12
  char v8; // al
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  char v12; // al
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // rax
  __int64 v16; // rax
  _QWORD *v17; // rdx
  __int64 v18; // rax
  _QWORD *v19; // rdx

  v4 = *(_BYTE *)(a2 + 16);
  if ( v4 == 50 )
  {
    v6 = *(_QWORD *)(a2 - 48);
    if ( !v6 )
      return 0;
    **a1 = v6;
    v7 = *(_QWORD *)(a2 - 24);
    v8 = *(_BYTE *)(v7 + 16);
    if ( v8 != 52 )
    {
      if ( v8 != 5 || *(_WORD *)(v7 + 18) != 28 )
        return 0;
      v9 = *(_DWORD *)(v7 + 20) & 0xFFFFFFF;
      a4 = 4 * v9;
      v10 = *(_QWORD *)(v7 - 24 * v9);
      if ( !v10 )
        goto LABEL_19;
      *a1[1] = v10;
      if ( !sub_1757E30(
              *(_BYTE **)(v7 + 24 * (1LL - (*(_DWORD *)(v7 + 20) & 0xFFFFFFF))),
              a2,
              *(_DWORD *)(v7 + 20) & 0xFFFFFFF,
              a4) )
      {
LABEL_18:
        v9 = *(_DWORD *)(v7 + 20) & 0xFFFFFFF;
LABEL_19:
        v15 = *(_QWORD *)(v7 + 24 * (1 - v9));
        if ( v15 )
        {
          *a1[1] = v15;
          return sub_1757E30(
                   *(_BYTE **)(v7 - 24LL * (*(_DWORD *)(v7 + 20) & 0xFFFFFFF)),
                   a2,
                   4LL * (*(_DWORD *)(v7 + 20) & 0xFFFFFFF),
                   a4);
        }
        return 0;
      }
      return 1;
    }
  }
  else
  {
    if ( v4 != 5 )
      return 0;
    if ( *(_WORD *)(a2 + 18) != 26 )
      return 0;
    v11 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
    if ( !v11 )
      return 0;
    a4 = 1;
    **a1 = v11;
    v7 = *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
    v12 = *(_BYTE *)(v7 + 16);
    if ( v12 != 52 )
    {
      if ( v12 != 5 || *(_WORD *)(v7 + 18) != 28 )
        return 0;
      v9 = *(_DWORD *)(v7 + 20) & 0xFFFFFFF;
      a2 = 4 * v9;
      v13 = *(_QWORD *)(v7 - 24 * v9);
      if ( !v13 )
        goto LABEL_19;
      *a1[1] = v13;
      v14 = 1LL - (*(_DWORD *)(v7 + 20) & 0xFFFFFFF);
      if ( !sub_1757E30(*(_BYTE **)(v7 + 24 * v14), a2, v13, v14) )
        goto LABEL_18;
      return 1;
    }
  }
  v16 = *(_QWORD *)(v7 - 48);
  if ( v16 )
  {
    v17 = a1[1];
    *v17 = v16;
    if ( (unsigned __int8)sub_1757CC0(*(_BYTE **)(v7 - 24), a2, (__int64)v17, a4) )
      return 1;
  }
  v18 = *(_QWORD *)(v7 - 24);
  if ( !v18 )
    return 0;
  v19 = a1[1];
  *v19 = v18;
  return sub_1757CC0(*(_BYTE **)(v7 - 48), a2, (__int64)v19, a4);
}
