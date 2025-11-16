// Function: sub_1B6E190
// Address: 0x1b6e190
//
__int64 __fastcall sub_1B6E190(__int64 a1, unsigned __int16 *a2, unsigned __int16 *a3, unsigned __int64 a4, __int64 a5)
{
  __int64 v5; // r13
  unsigned __int16 *v8; // rbx
  __int64 v9; // rbx
  __int64 v10; // rax
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // r12
  unsigned __int64 v13; // rax
  __int64 v14; // r12
  __int64 v15; // rax
  __int64 result; // rax
  __int64 v17; // rsi
  __int64 v18; // rcx
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // [rsp+8h] [rbp-58h]
  __int64 v22; // [rsp+18h] [rbp-48h]
  __int64 v23; // [rsp+20h] [rbp-40h] BYREF
  __int64 v24[7]; // [rsp+28h] [rbp-38h] BYREF

  v5 = a1;
  if ( (unsigned __int8)(*(_BYTE *)(a1 + 8) - 13) <= 1u )
    return 0xFFFFFFFFLL;
  v23 = 0;
  v24[0] = 0;
  v8 = sub_14AC610(a3, &v23, a5);
  if ( v8 != sub_14AC610(a2, v24, a5) )
    return 0xFFFFFFFFLL;
  v9 = 1;
  while ( 2 )
  {
    switch ( *(_BYTE *)(v5 + 8) )
    {
      case 0:
      case 8:
      case 0xA:
      case 0xC:
      case 0x10:
        v19 = *(_QWORD *)(v5 + 32);
        v5 = *(_QWORD *)(v5 + 24);
        v9 *= v19;
        continue;
      case 1:
        v10 = 16;
        goto LABEL_6;
      case 2:
        v10 = 32;
        goto LABEL_6;
      case 3:
      case 9:
        v10 = 64;
        goto LABEL_6;
      case 4:
        v10 = 80;
        goto LABEL_6;
      case 5:
      case 6:
        v10 = 128;
        goto LABEL_6;
      case 7:
        v10 = 8 * (unsigned int)sub_15A9520(a5, 0);
        goto LABEL_6;
      case 0xB:
        v10 = *(_DWORD *)(v5 + 8) >> 8;
        goto LABEL_6;
      case 0xD:
        v10 = 8LL * *(_QWORD *)sub_15A9930(a5, v5);
        goto LABEL_6;
      case 0xE:
        v22 = *(_QWORD *)(v5 + 24);
        sub_15A9FE0(a5, v22);
        v17 = v22;
        v18 = 1;
        while ( 2 )
        {
          switch ( *(_BYTE *)(v17 + 8) )
          {
            case 0:
            case 8:
            case 0xA:
            case 0xC:
            case 0x10:
              v20 = *(_QWORD *)(v17 + 32);
              v17 = *(_QWORD *)(v17 + 24);
              v18 *= v20;
              continue;
            case 1:
            case 2:
            case 3:
            case 4:
            case 5:
            case 6:
            case 9:
            case 0xB:
              goto LABEL_27;
            case 7:
              sub_15A9520(a5, 0);
              goto LABEL_27;
            case 0xD:
              sub_15A9930(a5, v17);
              goto LABEL_27;
            case 0xE:
              v21 = *(_QWORD *)(v17 + 24);
              sub_15A9FE0(a5, v21);
              sub_127FA20(a5, v21);
              JUMPOUT(0x1B6E458);
            case 0xF:
              sub_15A9520(a5, *(_DWORD *)(v17 + 8) >> 8);
LABEL_27:
              JUMPOUT(0x1B6E3AF);
          }
        }
      case 0xF:
        v10 = 8 * (unsigned int)sub_15A9520(a5, *(_DWORD *)(v5 + 8) >> 8);
LABEL_6:
        v11 = v9 * v10;
        if ( (((unsigned __int8)v11 | (unsigned __int8)a4) & 7) != 0 )
          return 0xFFFFFFFFLL;
        v12 = a4 >> 3;
        v13 = v11 >> 3;
        if ( v23 >= v24[0] )
        {
          v15 = v24[0] + v13;
          if ( v23 >= v15 || v23 > v24[0] )
            return 0xFFFFFFFFLL;
          v14 = v23 + v12;
        }
        else
        {
          v14 = v23 + v12;
          if ( v24[0] >= v14 )
            return 0xFFFFFFFFLL;
          v15 = v24[0] + v13;
        }
        if ( v14 < (unsigned __int64)v15 )
          return 0xFFFFFFFFLL;
        result = (unsigned int)(LODWORD(v24[0]) - v23);
        break;
    }
    return result;
  }
}
