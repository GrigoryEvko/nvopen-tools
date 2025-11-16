// Function: sub_1184DC0
// Address: 0x1184dc0
//
__int64 __fastcall sub_1184DC0(_QWORD **a1, __int64 a2)
{
  __int64 v3; // rdi
  unsigned int v4; // eax
  unsigned int v5; // r12d
  __int64 v6; // r14
  _BYTE *v8; // rdi
  unsigned __int8 *v9; // r15
  __int64 v10; // rbx
  char v11; // al
  __int64 v12; // rsi
  __int64 v13; // rax

  if ( *(_BYTE *)a2 <= 0x1Cu )
    return 0;
  v3 = *(_QWORD *)(a2 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v3 + 8) - 17 <= 1 )
    v3 = **(_QWORD **)(v3 + 16);
  LOBYTE(v4) = sub_BCAC40(v3, 1);
  v5 = v4;
  if ( !(_BYTE)v4 )
    return 0;
  if ( *(_BYTE *)a2 == 57 )
  {
    if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
      v10 = *(_QWORD *)(a2 - 8);
    else
      v10 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
    v9 = *(unsigned __int8 **)v10;
    v6 = *(_QWORD *)(v10 + 32);
    if ( **(_BYTE **)v10 == 59 )
    {
      v11 = sub_995B10(a1, *((_QWORD *)v9 - 8));
      v12 = *((_QWORD *)v9 - 4);
      if ( v11 && v12 )
      {
        *a1[1] = v12;
      }
      else
      {
        if ( !(unsigned __int8)sub_995B10(a1, v12) )
          goto LABEL_18;
        v13 = *((_QWORD *)v9 - 8);
        if ( !v13 )
          goto LABEL_18;
        *a1[1] = v13;
      }
      if ( v6 )
        goto LABEL_14;
    }
LABEL_18:
    v5 = sub_996420(a1, 30, (unsigned __int8 *)v6);
    if ( !(_BYTE)v5 )
      return 0;
LABEL_19:
    *a1[2] = v9;
    return v5;
  }
  if ( *(_BYTE *)a2 != 86 )
    return 0;
  v6 = *(_QWORD *)(a2 - 96);
  if ( *(_QWORD *)(a2 + 8) != *(_QWORD *)(v6 + 8) )
    return 0;
  v8 = *(_BYTE **)(a2 - 32);
  if ( *v8 > 0x15u )
    return 0;
  v9 = *(unsigned __int8 **)(a2 - 64);
  if ( !sub_AC30F0((__int64)v8) )
    return 0;
  LOBYTE(v5) = sub_996420(a1, 30, (unsigned __int8 *)v6) & (v9 != 0);
  if ( (_BYTE)v5 )
    goto LABEL_19;
  v5 = sub_996420(a1, 30, v9);
  if ( (_BYTE)v5 )
  {
LABEL_14:
    *a1[2] = v6;
    return v5;
  }
  return 0;
}
