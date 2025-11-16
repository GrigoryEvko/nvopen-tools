// Function: sub_100ADD0
// Address: 0x100add0
//
__int64 __fastcall sub_100ADD0(_QWORD **a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r13
  _BYTE *v8; // rdi
  _BYTE *v9; // rbx
  __int64 result; // rax
  __int64 v11; // rbx
  _BYTE *v12; // r14
  _BYTE *v13; // rbx
  __int64 v14; // rsi
  __int64 v15; // rdx

  if ( *(_BYTE *)a2 <= 0x1Cu )
    return 0;
  v3 = *(_QWORD *)(a2 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v3 + 8) - 17 <= 1 )
    v3 = **(_QWORD **)(v3 + 16);
  if ( !sub_BCAC40(v3, 1) )
    return 0;
  if ( *(_BYTE *)a2 == 58 )
  {
    if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
      v11 = *(_QWORD *)(a2 - 8);
    else
      v11 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
    v12 = *(_BYTE **)v11;
    v13 = *(_BYTE **)(v11 + 32);
    if ( v12 )
    {
      **a1 = v12;
      if ( *v13 == 59 )
      {
        result = sub_995B10(a1 + 1, *((_QWORD *)v13 - 8));
        v14 = *((_QWORD *)v13 - 4);
        if ( (_BYTE)result && v14 )
          goto LABEL_19;
        result = sub_995B10(a1 + 1, v14);
        if ( (_BYTE)result )
        {
          v15 = *((_QWORD *)v13 - 8);
          if ( v15 )
            goto LABEL_29;
        }
      }
    }
    else if ( !v13 )
    {
      return 0;
    }
    **a1 = v13;
    if ( *v12 == 59 )
    {
      result = sub_995B10(a1 + 1, *((_QWORD *)v12 - 8));
      v14 = *((_QWORD *)v12 - 4);
      if ( !(_BYTE)result || !v14 )
      {
        result = sub_995B10(a1 + 1, v14);
        if ( !(_BYTE)result )
          return 0;
        v15 = *((_QWORD *)v12 - 8);
        if ( !v15 )
          return 0;
        goto LABEL_29;
      }
LABEL_19:
      *a1[2] = v14;
      return result;
    }
    return 0;
  }
  if ( *(_BYTE *)a2 != 86 )
    return 0;
  v7 = *(_QWORD *)(a2 - 96);
  if ( *(_QWORD *)(a2 + 8) != *(_QWORD *)(v7 + 8) )
    return 0;
  v8 = *(_BYTE **)(a2 - 64);
  if ( *v8 > 0x15u )
    return 0;
  v9 = *(_BYTE **)(a2 - 32);
  if ( !sub_AD7A80(v8, 1, v4, v5, v6) )
    return 0;
  **a1 = v7;
  if ( *v9 != 59 )
    goto LABEL_24;
  result = sub_995B10(a1 + 1, *((_QWORD *)v9 - 8));
  v14 = *((_QWORD *)v9 - 4);
  if ( (_BYTE)result && v14 )
    goto LABEL_19;
  result = sub_995B10(a1 + 1, v14);
  if ( !(_BYTE)result || (v15 = *((_QWORD *)v9 - 8)) == 0 )
  {
LABEL_24:
    **a1 = v9;
    return sub_996420(a1 + 1, 30, (unsigned __int8 *)v7);
  }
LABEL_29:
  *a1[2] = v15;
  return result;
}
