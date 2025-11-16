// Function: sub_1181750
// Address: 0x1181750
//
bool __fastcall sub_1181750(_QWORD **a1, __int64 a2)
{
  __int64 v2; // rax
  bool result; // al
  __int64 v5; // rdx
  __int64 v6; // r13
  __int64 v7; // rcx
  __int64 v8; // r15
  int v9; // eax
  int v10; // edi

  if ( !a2 )
    return 0;
  v2 = *(_QWORD *)(a2 - 96);
  if ( *(_BYTE *)v2 != 83 )
    return 0;
  v5 = *(_QWORD *)(a2 - 64);
  v6 = *(_QWORD *)(v2 - 64);
  v7 = *(_QWORD *)(a2 - 32);
  v8 = *(_QWORD *)(v2 - 32);
  if ( v5 == v6 && v7 == v8 )
  {
    if ( (*(_WORD *)(v2 + 2) & 0x3Fu) - 2 > 1 || !v6 )
      goto LABEL_8;
    goto LABEL_17;
  }
  if ( v5 != v8 || v7 != v6 )
    goto LABEL_21;
  v10 = *(_WORD *)(v2 + 2) & 0x3F;
  if ( v5 == v6 )
  {
    if ( (unsigned int)(v10 - 2) > 1 || !v5 )
      goto LABEL_21;
    goto LABEL_17;
  }
  if ( (unsigned int)sub_B52870(v10) - 2 <= 1 && v6 )
  {
LABEL_17:
    **a1 = v6;
    if ( v8 )
    {
      *a1[1] = v8;
      return 1;
    }
  }
  v2 = *(_QWORD *)(a2 - 96);
  if ( *(_BYTE *)v2 != 83 )
    return 0;
  v7 = *(_QWORD *)(a2 - 32);
  v8 = *(_QWORD *)(v2 - 32);
  v5 = *(_QWORD *)(a2 - 64);
  v6 = *(_QWORD *)(v2 - 64);
  if ( v7 == v8 && v6 == v5 )
    goto LABEL_8;
LABEL_21:
  if ( v8 == v5 && v7 == v6 )
  {
    if ( v6 != v5 )
    {
      v9 = sub_B52870(*(_WORD *)(v2 + 2) & 0x3F);
      goto LABEL_9;
    }
LABEL_8:
    v9 = *(_WORD *)(v2 + 2) & 0x3F;
LABEL_9:
    result = v6 != 0 && (unsigned int)(v9 - 10) <= 1;
    if ( result )
    {
      *a1[2] = v6;
      if ( v8 )
      {
        *a1[3] = v8;
        return result;
      }
    }
  }
  return 0;
}
