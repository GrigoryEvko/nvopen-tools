// Function: sub_1C30010
// Address: 0x1c30010
//
__int64 __fastcall sub_1C30010(__int64 a1, unsigned int a2, _DWORD *a3)
{
  __int64 v5; // rax
  int v6; // ecx
  unsigned __int64 v7; // rsi
  __int64 v8; // rdx
  __int64 v9; // rcx
  _QWORD *v10; // rax
  unsigned __int16 v11; // cx
  unsigned int v12; // eax

  if ( !*(_QWORD *)(a1 + 48) && *(__int16 *)(a1 + 18) >= 0 )
    return 0;
  v5 = sub_1625940(a1, "callalign", 9u);
  if ( !v5 )
    return 0;
  v6 = *(_DWORD *)(v5 + 8);
  if ( v6 <= 0 )
    return 0;
  v7 = v5 + 8 * ((unsigned int)(v6 - 1) - (unsigned __int64)(unsigned int)v6) + 8;
  v8 = v5 - 8LL * (unsigned int)v6;
  while ( 1 )
  {
    v9 = *(_QWORD *)(*(_QWORD *)v8 + 136LL);
    if ( v9 )
      break;
LABEL_10:
    v8 += 8;
    if ( v7 == v8 )
      return 0;
  }
  v10 = *(_QWORD **)(v9 + 24);
  if ( *(_DWORD *)(v9 + 32) > 0x40u )
    v10 = (_QWORD *)*v10;
  v11 = (unsigned __int16)v10;
  v12 = (unsigned int)v10 >> 16;
  if ( v12 != a2 )
  {
    if ( v12 > a2 )
      return 0;
    goto LABEL_10;
  }
  *a3 = v11;
  return 1;
}
