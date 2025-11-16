// Function: sub_1969460
// Address: 0x1969460
//
__int64 __fastcall sub_1969460(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // r12
  __int64 v6; // rdi
  unsigned int v7; // r14d
  int v8; // eax

  v2 = 0;
  v3 = *(_QWORD *)(a1 - 72);
  if ( *(_BYTE *)(v3 + 16) == 75 )
  {
    v6 = *(_QWORD *)(v3 - 24);
    if ( *(_BYTE *)(v6 + 16) == 13 )
    {
      v7 = *(_DWORD *)(v6 + 32);
      if ( v7 <= 0x40 )
      {
        if ( *(_QWORD *)(v6 + 24) )
          return v2;
      }
      else if ( v7 != (unsigned int)sub_16A57B0(v6 + 24) )
      {
        return v2;
      }
      v8 = *(unsigned __int16 *)(v3 + 18);
      BYTE1(v8) &= ~0x80u;
      if ( v8 == 33 )
      {
        v2 = 0;
        if ( *(_QWORD *)(a1 - 24) != a2 )
          return v2;
        return *(_QWORD *)(v3 - 48);
      }
      v2 = 0;
      if ( v8 == 32 && a2 == *(_QWORD *)(a1 - 48) )
        return *(_QWORD *)(v3 - 48);
    }
  }
  return v2;
}
