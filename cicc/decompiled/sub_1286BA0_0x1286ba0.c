// Function: sub_1286BA0
// Address: 0x1286ba0
//
__int64 __fastcall sub_1286BA0(__int64 a1, _QWORD *a2, __int64 a3)
{
  unsigned __int8 v3; // al
  unsigned __int64 v4; // r13
  __int64 v6; // r14
  int v7; // ebx
  _BOOL4 v8; // edx
  __int64 v9; // [rsp+10h] [rbp-60h] BYREF
  int v10; // [rsp+18h] [rbp-58h]
  int v11; // [rsp+20h] [rbp-50h]
  __int64 v12; // [rsp+30h] [rbp-40h] BYREF
  int v13; // [rsp+38h] [rbp-38h]
  int v14; // [rsp+40h] [rbp-30h]

  v3 = *(_BYTE *)(a3 + 56);
  v4 = *(_QWORD *)a3;
  if ( v3 > 0x70u )
    goto LABEL_13;
  if ( v3 <= 0x48u )
  {
    if ( v3 != 8 )
    {
      if ( v3 > 8u )
      {
        if ( v3 == 25 )
        {
          sub_1287390();
          return a1;
        }
LABEL_13:
        sub_127B550("unsupported expression kind!", (_DWORD *)(a3 + 36), 1);
      }
      if ( v3 == 3 )
      {
        sub_1280BE0(a1, (__int64)a2, a3);
        return a1;
      }
      if ( v3 != 6 )
        goto LABEL_13;
    }
    sub_12873B0(a1);
    return a1;
  }
  switch ( v3 )
  {
    case 'I':
    case '[':
      sub_1287240();
      return a1;
    case '\\':
      sub_12865B0(a1, a2, a3);
      return a1;
    case '^':
      sub_1286E40();
      return a1;
    case '_':
      sub_1280A40(a1, a2, a3);
      return a1;
    case 'g':
      sub_1280D20(a1, a2, (unsigned __int64 *)a3);
      return a1;
    case 'i':
      sub_1281200((__int64)&v9);
      v6 = v9;
      if ( !v10 )
        goto LABEL_19;
      v7 = v11;
      goto LABEL_24;
    case 'p':
      sub_1280010((__int64)&v12, a2, (__int64 *)a3, 0);
      v6 = v12;
      if ( v13 )
      {
        v7 = v14;
LABEL_24:
        v8 = 0;
        if ( (*(_BYTE *)(v4 + 140) & 0xFB) == 8 )
          v8 = (sub_8D4C10(v4, dword_4F077C4 != 2) & 2) != 0;
        *(_DWORD *)a1 = 0;
        *(_QWORD *)(a1 + 8) = v6;
        *(_DWORD *)(a1 + 40) = v8;
        *(_DWORD *)(a1 + 16) = v7;
      }
      else
      {
LABEL_19:
        sub_12800D0(a1, a2, v4, v6);
      }
      break;
    default:
      goto LABEL_13;
  }
  return a1;
}
