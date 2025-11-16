// Function: sub_1705480
// Address: 0x1705480
//
__int64 __fastcall sub_1705480(double a1, double a2, double a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v7; // rcx
  char v8; // al
  unsigned int v10; // ebx
  int v11; // r13d
  __int64 v12; // rax

  if ( sub_15FB6B0(a5, a5, a6, a7) )
    return sub_15FB790(a5);
  v8 = *(_BYTE *)(a5 + 16);
  switch ( v8 )
  {
    case 13:
      return sub_15A2B90((__int64 *)a5, 0, 0, v7, a1, a2, a3);
    case 12:
      if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)a5 + 24LL) + 8LL) == 11 )
        return sub_15A2B90((__int64 *)a5, 0, 0, v7, a1, a2, a3);
      break;
    case 8:
      v10 = 0;
      v11 = *(_DWORD *)(a5 + 20) & 0xFFFFFFF;
      if ( !v11 )
        return sub_15A2B90((__int64 *)a5, 0, 0, v7, a1, a2, a3);
      while ( 1 )
      {
        v12 = sub_15A0A60(a5, v10);
        if ( !v12 || (*(_BYTE *)(v12 + 16) & 0xFB) != 9 )
          break;
        if ( ++v10 == v11 )
          return sub_15A2B90((__int64 *)a5, 0, 0, v7, a1, a2, a3);
      }
      break;
  }
  return 0;
}
