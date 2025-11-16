// Function: sub_11580F0
// Address: 0x11580f0
//
__int64 __fastcall sub_11580F0(_QWORD **a1, char *a2)
{
  __int64 v2; // rax
  char v5; // al
  __int64 v6; // rax
  _BYTE *v7; // rdi
  __int64 v8; // rax
  _BYTE *v9; // rdi

  v2 = *((_QWORD *)a2 + 2);
  if ( !v2 || *(_QWORD *)(v2 + 8) )
    return 0;
  v5 = *a2;
  if ( *a2 != 42 )
  {
LABEL_5:
    if ( v5 != 58 )
      return 0;
    if ( (a2[1] & 2) == 0 )
      return 0;
    v6 = *((_QWORD *)a2 - 8);
    if ( !v6 )
      return 0;
    *a1[3] = v6;
    v7 = (_BYTE *)*((_QWORD *)a2 - 4);
    if ( *v7 > 0x15u )
      return 0;
    *a1[4] = v7;
    if ( *v7 <= 0x15u )
    {
      if ( *v7 != 5 )
        return (unsigned int)sub_AD6CA0((__int64)v7) ^ 1;
      return 0;
    }
    return 1;
  }
  v8 = *((_QWORD *)a2 - 8);
  if ( !v8 )
    return 0;
  **a1 = v8;
  v9 = (_BYTE *)*((_QWORD *)a2 - 4);
  if ( *v9 > 0x15u || (*a1[1] = v9, *v9 <= 0x15u) && (*v9 == 5 || (unsigned __int8)sub_AD6CA0((__int64)v9)) )
  {
    v5 = *a2;
    goto LABEL_5;
  }
  return 1;
}
