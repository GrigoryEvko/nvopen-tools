// Function: sub_15AAE60
// Address: 0x15aae60
//
__int64 __fastcall sub_15AAE60(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  unsigned int v3; // eax
  unsigned int v4; // ebx
  unsigned int v5; // r14d
  unsigned int v7; // eax
  __int64 v8; // rbx
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // r15
  unsigned __int64 v12; // r13
  __int64 v13; // [rsp+8h] [rbp-38h]

  v2 = *(_QWORD *)(a2 + 24);
  v3 = sub_15AAE50(a1, v2);
  v4 = (unsigned int)(1 << (*(_DWORD *)(a2 + 32) >> 15)) >> 1;
  if ( v3 <= v4 )
  {
    v5 = (unsigned int)(1 << (*(_DWORD *)(a2 + 32) >> 15)) >> 1;
    goto LABEL_3;
  }
  v5 = v3;
  if ( !v4 )
  {
LABEL_3:
    if ( ((v5 <= 0xF) & ((unsigned __int8)sub_15E4F60(a2) ^ 1)) != 0
      && !v4
      && *(_DWORD *)(*(_QWORD *)a2 + 8LL) >> 8 != 3 )
    {
      v8 = 1;
      while ( 2 )
      {
        switch ( *(_BYTE *)(v2 + 8) )
        {
          case 0:
          case 8:
          case 0xA:
          case 0xC:
          case 0x10:
            v10 = *(_QWORD *)(v2 + 32);
            v2 = *(_QWORD *)(v2 + 24);
            v8 *= v10;
            continue;
          case 1:
            v9 = 16;
            break;
          case 2:
            v9 = 32;
            break;
          case 3:
          case 9:
            v9 = 64;
            break;
          case 4:
            v9 = 80;
            break;
          case 5:
          case 6:
            v9 = 128;
            break;
          case 7:
            v9 = 8 * (unsigned int)sub_15A9520(a1, 0);
            break;
          case 0xB:
            v9 = *(_DWORD *)(v2 + 8) >> 8;
            break;
          case 0xD:
            v9 = 8LL * *(_QWORD *)sub_15A9930(a1, v2);
            break;
          case 0xE:
            v11 = *(_QWORD *)(v2 + 32);
            v13 = *(_QWORD *)(v2 + 24);
            v12 = (unsigned int)sub_15A9FE0(a1, v13);
            v9 = 8 * v12 * v11 * ((v12 + ((unsigned __int64)(sub_127FA20(a1, v13) + 7) >> 3) - 1) / v12);
            break;
          case 0xF:
            v9 = 8 * (unsigned int)sub_15A9520(a1, *(_DWORD *)(v2 + 8) >> 8);
            break;
        }
        break;
      }
      if ( (unsigned __int64)(v9 * v8) >= 0x81 )
        return 16;
    }
    return v5;
  }
  v7 = sub_15A9FE0(a1, v2);
  if ( v7 >= v4 )
    v4 = v7;
  v5 = v4;
  sub_15E4F60(a2);
  return v5;
}
