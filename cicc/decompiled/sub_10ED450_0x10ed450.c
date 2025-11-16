// Function: sub_10ED450
// Address: 0x10ed450
//
__int64 __fastcall sub_10ED450(_QWORD **a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r13
  __int64 v4; // r12
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int16 v11; // ax
  int v12; // eax
  __int64 v13; // rax
  __int64 v14; // rax
  char *v15; // rsi

  if ( !a2 )
    goto LABEL_55;
  v2 = *(_QWORD *)(a2 - 32);
  if ( !v2 || *(_BYTE *)v2 || *(_QWORD *)(v2 + 24) != *(_QWORD *)(a2 + 80) )
    BUG();
  if ( *(_DWORD *)(v2 + 36) != 329 )
  {
LABEL_55:
    if ( *(_BYTE *)a2 != 86 )
      return 0;
    v8 = *(_QWORD *)(a2 - 96);
    if ( *(_BYTE *)v8 != 82 )
      return 0;
    v9 = *(_QWORD *)(a2 - 64);
    v4 = *(_QWORD *)(v8 - 64);
    v10 = *(_QWORD *)(a2 - 32);
    v3 = *(_QWORD *)(v8 - 32);
    if ( v4 == v9 && v3 == v10 )
    {
      v11 = *(_WORD *)(v8 + 2);
    }
    else
    {
      if ( v3 != v9 || v4 != v10 )
        return 0;
      v11 = *(_WORD *)(v8 + 2);
      if ( v4 != v9 )
      {
        v12 = sub_B52870(v11 & 0x3F);
        goto LABEL_16;
      }
    }
    v12 = v11 & 0x3F;
LABEL_16:
    if ( (unsigned int)(v12 - 38) > 1 )
      return 0;
    v13 = *(_QWORD *)(v4 + 16);
    if ( v13 )
    {
      if ( !*(_QWORD *)(v13 + 8) && *(_BYTE *)v4 > 0x1Cu )
      {
        **a1 = v4;
        if ( (unsigned __int8)sub_10ECBA0(a1 + 1, (char *)v4)
          || (unsigned __int8)sub_10ECD10(a1 + 4, (char *)v4)
          || (unsigned __int8)sub_10ECE80(a1 + 7, (char *)v4)
          || (unsigned __int8)sub_10ECFF0(a1 + 10, (char *)v4) )
        {
          if ( v3 )
            goto LABEL_43;
        }
      }
    }
    v14 = *(_QWORD *)(v3 + 16);
    if ( !v14 )
      return 0;
    if ( *(_QWORD *)(v14 + 8) )
      return 0;
    if ( *(_BYTE *)v3 <= 0x1Cu )
      return 0;
    **a1 = v3;
    if ( !(unsigned __int8)sub_10ECBA0(a1 + 1, (char *)v3)
      && !(unsigned __int8)sub_10ECD10(a1 + 4, (char *)v3)
      && !(unsigned __int8)sub_10ECE80(a1 + 7, (char *)v3)
      && !(unsigned __int8)sub_10ECFF0(a1 + 10, (char *)v3) )
    {
      return 0;
    }
LABEL_25:
    *a1[13] = v4;
    return 1;
  }
  v3 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v4 = *(_QWORD *)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
  v5 = *(_QWORD *)(v3 + 16);
  if ( v5 )
  {
    if ( !*(_QWORD *)(v5 + 8) && *(_BYTE *)v3 > 0x1Cu )
    {
      v15 = *(char **)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
      **a1 = v3;
      if ( (unsigned __int8)sub_10ECBA0(a1 + 1, v15)
        || (unsigned __int8)sub_10ECD10(a1 + 4, (char *)v3)
        || (unsigned __int8)sub_10ECE80(a1 + 7, (char *)v3)
        || (unsigned __int8)sub_10ECFF0(a1 + 10, (char *)v3) )
      {
        if ( v4 )
          goto LABEL_25;
      }
    }
  }
  v6 = *(_QWORD *)(v4 + 16);
  if ( !v6 )
    return 0;
  if ( *(_QWORD *)(v6 + 8) )
    return 0;
  if ( *(_BYTE *)v4 <= 0x1Cu )
    return 0;
  **a1 = v4;
  if ( !(unsigned __int8)sub_10ECBA0(a1 + 1, (char *)v4)
    && !(unsigned __int8)sub_10ECD10(a1 + 4, (char *)v4)
    && !(unsigned __int8)sub_10ECE80(a1 + 7, (char *)v4)
    && !(unsigned __int8)sub_10ECFF0(a1 + 10, (char *)v4) )
  {
    return 0;
  }
LABEL_43:
  *a1[13] = v3;
  return 1;
}
