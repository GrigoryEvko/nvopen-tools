// Function: sub_1593BB0
// Address: 0x1593bb0
//
bool __fastcall sub_1593BB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rdx
  unsigned int v5; // r12d
  bool result; // al
  __int64 v7; // rbx
  __int64 v8; // rbx

  v4 = *(unsigned __int8 *)(a1 + 16);
  if ( (_BYTE)v4 != 13 )
  {
    if ( (_BYTE)v4 != 14 )
      return (_BYTE)v4 == 10 || (unsigned __int8)(v4 - 15) <= 1u;
    if ( *(_QWORD *)(a1 + 32) == sub_16982C0(a1, a2, v4, a4) )
    {
      v8 = *(_QWORD *)(a1 + 40);
      result = 0;
      if ( (*(_BYTE *)(v8 + 26) & 7) != 3 )
        return result;
      v7 = v8 + 8;
    }
    else
    {
      result = 0;
      v7 = a1 + 32;
      if ( (*(_BYTE *)(a1 + 50) & 7) != 3 )
        return result;
    }
    return ((*(_BYTE *)(v7 + 18) >> 3) ^ 1) & 1;
  }
  v5 = *(_DWORD *)(a1 + 32);
  if ( v5 <= 0x40 )
    return *(_QWORD *)(a1 + 24) == 0;
  else
    return v5 == (unsigned int)sub_16A57B0(a1 + 24);
}
