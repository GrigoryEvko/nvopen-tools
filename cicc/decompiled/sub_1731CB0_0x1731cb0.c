// Function: sub_1731CB0
// Address: 0x1731cb0
//
__int64 __fastcall sub_1731CB0(_QWORD **a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned int v3; // r8d
  int v5; // eax
  int v6; // eax
  _QWORD *v7; // rsi

  v2 = *(_QWORD *)(a2 + 8);
  v3 = 0;
  if ( v2 && !*(_QWORD *)(v2 + 8) )
  {
    v5 = *(unsigned __int8 *)(a2 + 16);
    if ( (unsigned __int8)v5 <= 0x17u )
    {
      if ( (_BYTE)v5 != 5 )
        return v3;
      v6 = *(unsigned __int16 *)(a2 + 18);
    }
    else
    {
      v6 = v5 - 24;
    }
    v3 = 0;
    if ( v6 == 38 )
    {
      if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
        v7 = *(_QWORD **)(a2 - 8);
      else
        v7 = (_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
      v3 = 0;
      if ( *v7 )
      {
        v3 = 1;
        **a1 = *v7;
      }
    }
  }
  return v3;
}
