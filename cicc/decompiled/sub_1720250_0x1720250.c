// Function: sub_1720250
// Address: 0x1720250
//
__int64 __fastcall sub_1720250(_QWORD **a1, __int64 a2)
{
  int v2; // eax
  unsigned int v3; // r8d
  int v5; // eax
  __int64 *v6; // rsi
  __int64 v7; // rdx
  int v8; // eax
  int v9; // eax
  _QWORD *v10; // rdx

  v2 = *(unsigned __int8 *)(a2 + 16);
  if ( (unsigned __int8)v2 > 0x17u )
  {
    v5 = v2 - 24;
  }
  else
  {
    if ( (_BYTE)v2 != 5 )
      return 0;
    v5 = *(unsigned __int16 *)(a2 + 18);
  }
  v3 = 0;
  if ( v5 != 36 )
    return v3;
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v6 = *(__int64 **)(a2 - 8);
  else
    v6 = (__int64 *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v7 = *v6;
  v8 = *(unsigned __int8 *)(*v6 + 16);
  if ( (unsigned __int8)v8 > 0x17u )
  {
    v9 = v8 - 24;
  }
  else
  {
    if ( (_BYTE)v8 != 5 )
      return 0;
    v9 = *(unsigned __int16 *)(v7 + 18);
  }
  v3 = 0;
  if ( v9 == 45 )
  {
    if ( (*(_BYTE *)(v7 + 23) & 0x40) != 0 )
      v10 = *(_QWORD **)(v7 - 8);
    else
      v10 = (_QWORD *)(v7 - 24LL * (*(_DWORD *)(v7 + 20) & 0xFFFFFFF));
    v3 = 0;
    if ( *v10 )
    {
      v3 = 1;
      **a1 = *v10;
    }
  }
  return v3;
}
