// Function: sub_28F3780
// Address: 0x28f3780
//
unsigned __int8 *__fastcall sub_28F3780(__int64 a1, unsigned __int8 *a2)
{
  unsigned __int8 *v2; // rbx
  unsigned __int8 v3; // al
  unsigned __int8 *v4; // r12
  _BYTE *v6; // rcx
  __int64 v7; // rdx
  __int64 v8; // rax
  _BYTE *v9; // rax
  __int64 v10; // rdx
  __int64 v11; // r8
  __int64 v12; // rax
  __int64 v13; // rax
  unsigned __int8 *v14; // rax
  _BYTE *v15; // rcx
  unsigned __int8 *v16; // rax

  v2 = a2;
  v3 = *a2;
  if ( *a2 != 43 )
  {
LABEL_2:
    v4 = v2;
    goto LABEL_3;
  }
  v10 = *((_QWORD *)a2 - 8);
  if ( !v10
    || (v11 = *((_QWORD *)a2 - 4), (v12 = *(_QWORD *)(v11 + 16)) == 0)
    || *(_QWORD *)(v12 + 8)
    || *(_BYTE *)v11 <= 0x1Cu )
  {
    v13 = *(_QWORD *)(v10 + 16);
    if ( v13 )
      goto LABEL_22;
    return v2;
  }
  v14 = sub_28F2E50(a1, a2, *((unsigned __int8 **)a2 - 4), *((_BYTE **)a2 - 8));
  if ( v14 )
    v2 = v14;
  v3 = *v2;
  if ( *v2 != 43 )
    goto LABEL_2;
  v10 = *((_QWORD *)v2 - 8);
  v13 = *(_QWORD *)(v10 + 16);
  if ( !v13 )
    return v2;
LABEL_22:
  if ( *(_QWORD *)(v13 + 8) )
    return v2;
  if ( *(_BYTE *)v10 <= 0x1Cu )
    return v2;
  v15 = (_BYTE *)*((_QWORD *)v2 - 4);
  if ( !v15 )
    return v2;
  v16 = sub_28F2E50(a1, v2, (unsigned __int8 *)v10, v15);
  v4 = v16;
  if ( v16 )
  {
    v3 = *v16;
  }
  else
  {
    v3 = *v2;
    v4 = v2;
  }
LABEL_3:
  if ( v3 != 45 )
    return v4;
  v6 = (_BYTE *)*((_QWORD *)v4 - 8);
  if ( !v6 )
    return v4;
  v7 = *((_QWORD *)v4 - 4);
  v8 = *(_QWORD *)(v7 + 16);
  if ( !v8 || *(_QWORD *)(v8 + 8) || *(_BYTE *)v7 <= 0x1Cu )
    return v4;
  v9 = sub_28F2E50(a1, v4, (unsigned __int8 *)v7, v6);
  if ( v9 )
    return v9;
  return v4;
}
