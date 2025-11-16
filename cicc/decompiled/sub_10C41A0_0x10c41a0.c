// Function: sub_10C41A0
// Address: 0x10c41a0
//
__int64 __fastcall sub_10C41A0(_QWORD **a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 result; // rax
  _BYTE *v4; // r12
  _BYTE *v6; // rdi
  __int64 v7; // rdx
  _BYTE *v8; // rdi

  v2 = *(_QWORD *)(a2 + 16);
  if ( !v2 || *(_QWORD *)(v2 + 8) || *(_BYTE *)a2 != 55 )
    return 0;
  v4 = *(_BYTE **)(a2 - 64);
  if ( *v4 != 54 )
    return 0;
  v6 = (_BYTE *)*((_QWORD *)v4 - 8);
  if ( *v6 > 0x15u )
    return 0;
  **a1 = v6;
  if ( *v6 <= 0x15u && (*v6 == 5 || (unsigned __int8)sub_AD6CA0((__int64)v6)) )
    return 0;
  v7 = *((_QWORD *)v4 - 4);
  if ( !v7 )
    return 0;
  *a1[2] = v7;
  v8 = *(_BYTE **)(a2 - 32);
  if ( *v8 > 0x15u )
    return 0;
  *a1[3] = v8;
  result = 1;
  if ( *v8 <= 0x15u )
  {
    if ( *v8 != 5 )
      return (unsigned int)sub_AD6CA0((__int64)v8) ^ 1;
    return 0;
  }
  return result;
}
