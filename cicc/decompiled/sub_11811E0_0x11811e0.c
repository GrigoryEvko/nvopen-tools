// Function: sub_11811E0
// Address: 0x11811e0
//
bool __fastcall sub_11811E0(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v4; // r13
  _BYTE *v5; // rdi
  __int64 v6; // rbx
  bool result; // al
  __int64 *v8; // rbx
  __int64 v9; // rdx
  __int64 v10; // rsi

  if ( *(_BYTE *)a2 <= 0x1Cu )
    return 0;
  v3 = *(_QWORD *)(a2 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v3 + 8) - 17 <= 1 )
    v3 = **(_QWORD **)(v3 + 16);
  if ( !sub_BCAC40(v3, 1) )
    return 0;
  if ( *(_BYTE *)a2 == 57 )
  {
    if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
      v8 = *(__int64 **)(a2 - 8);
    else
      v8 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
    v9 = v8[4];
    v10 = *v8;
    result = *v8 == *(_QWORD *)a1 && v9 != 0;
    if ( result )
    {
      **(_QWORD **)(a1 + 8) = v9;
      return result;
    }
    result = v9 == *(_QWORD *)a1 && v10 != 0;
    if ( result )
    {
      **(_QWORD **)(a1 + 8) = v10;
      return result;
    }
    return 0;
  }
  if ( *(_BYTE *)a2 != 86 )
    return 0;
  v4 = *(_QWORD *)(a2 - 96);
  if ( *(_QWORD *)(a2 + 8) != *(_QWORD *)(v4 + 8) )
    return 0;
  v5 = *(_BYTE **)(a2 - 32);
  if ( *v5 > 0x15u )
    return 0;
  v6 = *(_QWORD *)(a2 - 64);
  result = sub_AC30F0((__int64)v5);
  if ( !result )
    return 0;
  if ( v4 != *(_QWORD *)a1 )
  {
    if ( v6 == *(_QWORD *)a1 )
    {
      **(_QWORD **)(a1 + 8) = v4;
      return result;
    }
    return 0;
  }
  if ( !v6 )
    return 0;
  **(_QWORD **)(a1 + 8) = v6;
  return result;
}
