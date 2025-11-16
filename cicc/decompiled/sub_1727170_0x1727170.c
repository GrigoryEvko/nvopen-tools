// Function: sub_1727170
// Address: 0x1727170
//
__int64 __fastcall sub_1727170(_BYTE *a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v4; // eax
  unsigned int v5; // r12d
  unsigned __int8 v6; // al
  int v8; // r13d
  unsigned int v9; // r14d
  __int64 v10; // rax

  LOBYTE(v4) = sub_15FB730((__int64)a1, a2, a3, a4);
  if ( (_BYTE)v4 )
    return 1;
  v5 = v4;
  v6 = a1[16];
  if ( v6 == 13 )
    return 1;
  if ( *(_BYTE *)(*(_QWORD *)a1 + 8LL) == 16 && v6 <= 0x10u )
  {
    v8 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
    if ( v8 )
    {
      v9 = 0;
      while ( 1 )
      {
        v10 = sub_15A0A60((__int64)a1, v9);
        if ( !v10 || (*(_BYTE *)(v10 + 16) & 0xFB) != 9 )
          break;
        if ( v8 == ++v9 )
          return 1;
      }
      return v5;
    }
    return 1;
  }
  if ( v6 > 0x17u )
  {
    if ( (unsigned __int8)(v6 - 75) <= 1u )
      return (unsigned int)a2;
    if ( (unsigned int)v6 - 35 <= 0x11 && ((v6 - 35) & 0xFD) == 0 )
    {
      if ( *(_BYTE *)(*((_QWORD *)a1 - 6) + 16LL) <= 0x10u )
        return (unsigned int)a2;
      if ( *(_BYTE *)(*((_QWORD *)a1 - 3) + 16LL) <= 0x10u )
        return (unsigned int)a2;
    }
  }
  return v5;
}
