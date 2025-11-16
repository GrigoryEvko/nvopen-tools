// Function: sub_E6CA50
// Address: 0xe6ca50
//
_QWORD *__fastcall sub_E6CA50(__int64 a1, const void *a2, size_t a3)
{
  unsigned __int64 v3; // rax
  __int64 v4; // r14
  unsigned __int64 v5; // rbx
  __int64 v6; // rax
  _QWORD *v7; // rax
  _QWORD *v8; // r12
  unsigned __int64 v9; // rax
  __int64 v11; // rax
  const char *v12; // [rsp+0h] [rbp-50h] BYREF
  char v13; // [rsp+20h] [rbp-30h]
  char v14; // [rsp+21h] [rbp-2Fh]

  v3 = sub_E6B3F0(a1, a2, a3);
  v4 = *(_QWORD *)(v3 + 8);
  v5 = v3;
  if ( !v4 )
    goto LABEL_5;
  v6 = *(_QWORD *)v4;
  if ( *(_QWORD *)v4 )
  {
LABEL_3:
    if ( off_4C5D170 != (_UNKNOWN *)v6 && v4 == *(_QWORD *)(*(_QWORD *)(v6 + 8) + 16LL) )
      goto LABEL_5;
    v14 = 1;
    v12 = "invalid symbol redefinition";
    v13 = 3;
    sub_E66880(a1, 0, (__int64)&v12);
    if ( *(_QWORD *)v4 )
      goto LABEL_5;
LABEL_12:
    v8 = (_QWORD *)v4;
    if ( (*(_BYTE *)(v4 + 9) & 0x70) == 0x20 )
      goto LABEL_13;
    return v8;
  }
  v8 = (_QWORD *)v4;
  if ( (*(_BYTE *)(v4 + 9) & 0x70) != 0x20 )
    return v8;
  if ( *(char *)(v4 + 8) >= 0 )
  {
    *(_BYTE *)(v4 + 8) |= 8u;
    v6 = sub_E807D0(*(_QWORD *)(v4 + 24));
    *(_QWORD *)v4 = v6;
    if ( !v6 )
      goto LABEL_12;
    goto LABEL_3;
  }
LABEL_13:
  v8 = (_QWORD *)v4;
  if ( *(char *)(v4 + 8) < 0 )
    return v8;
  *(_BYTE *)(v4 + 8) |= 8u;
  v11 = sub_E807D0(*(_QWORD *)(v4 + 24));
  *(_QWORD *)v4 = v11;
  if ( !v11 )
    return v8;
LABEL_5:
  *(_BYTE *)(v5 + 20) = 1;
  v7 = (_QWORD *)sub_EA1530(40, v5, a1);
  v8 = v7;
  if ( v7 )
  {
    v7[3] = 0;
    *v7 = 0;
    v9 = v7[1] & 0xFFFF0000FFF00000LL | 0x201;
    *(v8 - 1) = v5;
    v8[1] = v9;
    *((_DWORD *)v8 + 4) = 0;
    v8[4] = 0;
  }
  if ( v4 )
    return v8;
  *(_QWORD *)(v5 + 8) = v8;
  return v8;
}
