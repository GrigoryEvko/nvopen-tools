// Function: sub_15FB960
// Address: 0x15fb960
//
__int64 __fastcall sub_15FB960(
        unsigned int a1,
        unsigned int a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  __int64 v8; // r13
  __int64 result; // rax
  unsigned int v10; // ebx
  unsigned int v11; // r13d
  unsigned int v12; // eax
  bool v13; // cf
  unsigned int v14; // r12d
  unsigned int v15; // ebx
  int v16; // eax
  __int64 v18; // [rsp+8h] [rbp-28h]
  __int64 v19; // [rsp+8h] [rbp-28h]

  v8 = a3;
  if ( (a1 != 47 || (*(_BYTE *)(a4 + 8) == 16) == (*(_BYTE *)(a3 + 8) == 16))
    && (a2 != 47 || (*(_BYTE *)(a5 + 8) == 16) == (*(_BYTE *)(a4 + 8) == 16))
    || a1 == 47 && a2 == 47 )
  {
    switch ( byte_42AB620[13 * (a1 - 36) + a2 - 36] )
    {
      case 0:
        return 0;
      case 1:
      case 0xF:
        return a1;
      case 2:
      case 0xE:
      case 0x10:
        return a2;
      case 3:
        result = 0;
        if ( *(_BYTE *)(a3 + 8) != 16 && *(_BYTE *)(a5 + 8) == 11 )
          return a1;
        return result;
      case 4:
        result = 0;
        if ( (unsigned __int8)(*(_BYTE *)(a5 + 8) - 1) <= 5u )
          return a1;
        return result;
      case 5:
        result = 0;
        if ( *(_BYTE *)(a3 + 8) == 11 )
          return a2;
        return result;
      case 6:
        result = 0;
        if ( (unsigned __int8)(*(_BYTE *)(a3 + 8) - 1) <= 5u )
          return a2;
        return result;
      case 7:
        if ( *(_BYTE *)(a3 + 8) == 16 )
          v8 = **(_QWORD **)(a3 + 16);
        if ( *(_BYTE *)(a5 + 8) == 16 )
          a5 = **(_QWORD **)(a5 + 16);
        result = 0;
        if ( *(_DWORD *)(a5 + 8) >> 8 != *(_DWORD *)(v8 + 8) >> 8 )
          return result;
        v10 = sub_16431D0(a4);
        if ( v10 == 64 )
          return 47;
        if ( a6 && a6 == a8 )
          return v10 >= (unsigned int)sub_16431D0(a6) ? 0x2F : 0;
        break;
      case 8:
        v18 = a5;
        v11 = sub_16431D0(a3);
        v12 = sub_16431D0(v18);
        v13 = v11 < v12;
        if ( v11 == v12 )
          return 47;
        result = a1;
        if ( !v13 )
          return a2;
        return result;
      case 9:
        return 37;
      case 0xA:
      case 0x11:
        return 41;
      case 0xB:
        v19 = a5;
        if ( !a7 )
          return 0;
        v14 = sub_16431D0(a7);
        v15 = sub_16431D0(v8);
        v16 = sub_16431D0(v19);
        if ( v14 < v15 || v15 != v16 )
          return 0;
        return 47;
      case 0xC:
        if ( *(_BYTE *)(a3 + 8) == 16 )
          v8 = **(_QWORD **)(a3 + 16);
        if ( *(_BYTE *)(a5 + 8) == 16 )
          a5 = **(_QWORD **)(a5 + 16);
        return (unsigned int)(*(_DWORD *)(a5 + 8) >> 8 != *(_DWORD *)(v8 + 8) >> 8) + 47;
      case 0xD:
        if ( *(_BYTE *)(a3 + 8) == 16 )
          v8 = **(_QWORD **)(a3 + 16);
        if ( *(_BYTE *)(a5 + 8) == 16 )
          a5 = **(_QWORD **)(a5 + 16);
        result = 48;
        if ( **(_QWORD **)(a5 + 16) != **(_QWORD **)(v8 + 16) )
          return 0;
        return result;
    }
  }
  return 0;
}
