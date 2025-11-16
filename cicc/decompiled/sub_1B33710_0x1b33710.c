// Function: sub_1B33710
// Address: 0x1b33710
//
__int64 __fastcall sub_1B33710(_QWORD *a1, char a2)
{
  __int64 v2; // r15
  int v3; // r13d
  char v4; // r14
  _QWORD *v5; // r8
  unsigned __int8 v6; // al
  _QWORD *v7; // rax
  _QWORD *v9; // rax
  __int64 v10; // rax
  _QWORD *v11; // rax
  __int64 v12; // [rsp+0h] [rbp-40h]
  __int64 v13; // [rsp+0h] [rbp-40h]
  __int64 v14; // [rsp+8h] [rbp-38h]
  __int64 v15; // [rsp+8h] [rbp-38h]

  v2 = a1[1];
  v3 = *(_DWORD *)(*a1 + 8LL) >> 8;
  if ( !v2 )
    return 1;
  v4 = 0;
  while ( 1 )
  {
    v5 = sub_1648700(v2);
    v6 = *((_BYTE *)v5 + 16);
    if ( v6 <= 0x17u )
      break;
    switch ( v6 )
    {
      case '6':
        if ( (*((_BYTE *)v5 + 18) & 1) != 0 )
          return 0;
        break;
      case '7':
        v9 = (_QWORD *)*(v5 - 6);
        if ( a1 == v9 && v9 || (*((_BYTE *)v5 + 18) & 1) != 0 )
          return 0;
        if ( a2 )
        {
          if ( *a1 && (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(*a1 + 24LL) + 8LL) - 13 <= 1 && v4 )
            return 0;
          v4 = a2;
        }
        else
        {
          v4 = 1;
        }
        break;
      case 'N':
        v10 = *(v5 - 3);
        if ( *(_BYTE *)(v10 + 16)
          || (*(_BYTE *)(v10 + 33) & 0x20) == 0
          || (unsigned int)(*(_DWORD *)(v10 + 36) - 116) > 1 )
        {
          return 0;
        }
        break;
      case 'G':
        v12 = (__int64)v5;
        v14 = *v5;
        v7 = (_QWORD *)sub_16498A0((__int64)v5);
        if ( v14 != sub_16471D0(v7, v3) || !(unsigned __int8)sub_14ADF20(v12) )
          return 0;
        break;
      case '8':
        v13 = (__int64)v5;
        v15 = *v5;
        v11 = (_QWORD *)sub_16498A0((__int64)v5);
        if ( v15 != sub_16471D0(v11, v3) || !(unsigned __int8)sub_15FA1F0(v13) || !(unsigned __int8)sub_14ADF20(v13) )
          return 0;
        break;
      default:
        if ( v6 != 72 || !(unsigned __int8)sub_14ADF20((__int64)v5) )
          return 0;
        break;
    }
    v2 = *(_QWORD *)(v2 + 8);
    if ( !v2 )
      return 1;
  }
  return 0;
}
