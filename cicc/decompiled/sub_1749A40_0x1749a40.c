// Function: sub_1749A40
// Address: 0x1749a40
//
__int64 __fastcall sub_1749A40(__int64 a1, _DWORD *a2, _QWORD *a3)
{
  unsigned __int64 v3; // rax
  _QWORD *v4; // rax
  __int64 result; // rax
  __int64 v6; // rcx
  __int64 v7; // r12
  _QWORD *v8; // rcx
  _QWORD *v9; // rcx
  _QWORD *v10; // rax
  _DWORD v12[5]; // [rsp+1Ch] [rbp-14h] BYREF

  v3 = *(unsigned __int8 *)(a1 + 16);
  if ( (_BYTE)v3 == 13 )
  {
    v4 = *(_QWORD **)(a1 + 24);
    if ( *(_DWORD *)(a1 + 32) > 0x40u )
      v4 = (_QWORD *)*v4;
    *a3 = v4;
    *a2 = 0;
    return sub_15A0680(*(_QWORD *)a1, 0, 0);
  }
  if ( (unsigned __int8)(v3 - 35) > 0x11u )
    goto LABEL_17;
  if ( (unsigned __int8)v3 > 0x2Fu )
    goto LABEL_17;
  v6 = 0x80A800000000LL;
  if ( !_bittest64(&v6, v3) || (*(_BYTE *)(a1 + 17) & 2) == 0 && ((*(_BYTE *)(a1 + 17) >> 1) & 2) == 0 )
    goto LABEL_17;
  v7 = *(_QWORD *)(a1 - 24);
  if ( *(_BYTE *)(v7 + 16) != 13 )
    goto LABEL_17;
  switch ( (_BYTE)v3 )
  {
    case '/':
      v9 = *(_QWORD **)(v7 + 24);
      if ( *(_DWORD *)(v7 + 32) > 0x40u )
        v9 = (_QWORD *)*v9;
      *a2 = 1LL << (char)v9;
      *a3 = 0;
      return *(_QWORD *)(a1 - 48);
    case '\'':
      v10 = *(_QWORD **)(v7 + 24);
      if ( *(_DWORD *)(v7 + 32) > 0x40u )
        v10 = (_QWORD *)*v10;
      *a2 = (_DWORD)v10;
      *a3 = 0;
      return *(_QWORD *)(a1 - 48);
    case '#':
      result = sub_1749A40(*(_QWORD *)(a1 - 48), v12);
      v8 = *(_QWORD **)(v7 + 24);
      if ( *(_DWORD *)(v7 + 32) > 0x40u )
        v8 = (_QWORD *)*v8;
      *a3 += v8;
      *a2 = v12[0];
      return result;
    default:
LABEL_17:
      *a2 = 1;
      *a3 = 0;
      return a1;
  }
}
