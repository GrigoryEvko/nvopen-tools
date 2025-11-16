// Function: sub_1CC52F0
// Address: 0x1cc52f0
//
unsigned __int64 __fastcall sub_1CC52F0(__int64 a1, unsigned int a2, int a3, __int64 a4)
{
  unsigned __int64 result; // rax
  __int64 v6; // r12
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // r15
  unsigned int v11; // eax
  __int64 v12; // rsi
  __int64 v13; // rcx
  unsigned __int64 v14; // r12
  __int64 v15; // rax
  __int64 v16; // rax
  _QWORD *v17; // rax
  int v18; // eax
  int v19; // eax
  __int64 v20; // rax
  __int64 v21; // [rsp+0h] [rbp-50h]
  __int64 v22; // [rsp+8h] [rbp-48h]
  unsigned __int64 v23; // [rsp+10h] [rbp-40h]
  __int64 v24; // [rsp+18h] [rbp-38h]
  __int64 v25; // [rsp+18h] [rbp-38h]
  __int64 v26; // [rsp+18h] [rbp-38h]
  __int64 v27; // [rsp+18h] [rbp-38h]
  __int64 v28; // [rsp+18h] [rbp-38h]

  result = a2;
  v6 = a4;
  if ( !a2 )
    result = sub_15AAE50(a1, a4);
  if ( (unsigned int)result <= 0xF && (((_BYTE)result - 1) & 0x10) == 0 )
  {
    v7 = 1;
    while ( 2 )
    {
      switch ( *(_BYTE *)(v6 + 8) )
      {
        case 0:
        case 8:
        case 0xA:
        case 0xC:
        case 0x10:
          v9 = *(_QWORD *)(v6 + 32);
          v6 = *(_QWORD *)(v6 + 24);
          v7 *= v9;
          continue;
        case 1:
          v8 = 16;
          break;
        case 2:
          v8 = 32;
          break;
        case 3:
        case 9:
          v8 = 64;
          break;
        case 4:
          v8 = 80;
          break;
        case 5:
        case 6:
          v8 = 128;
          break;
        case 7:
          v8 = 8 * (unsigned int)sub_15A9520(a1, 0);
          break;
        case 0xB:
          v8 = *(_DWORD *)(v6 + 8) >> 8;
          break;
        case 0xD:
          v8 = 8LL * *(_QWORD *)sub_15A9930(a1, v6);
          break;
        case 0xE:
          v10 = *(_QWORD *)(v6 + 32);
          v24 = *(_QWORD *)(v6 + 24);
          v11 = sub_15A9FE0(a1, v24);
          v12 = v24;
          v13 = 1;
          v14 = v11;
          while ( 2 )
          {
            switch ( *(_BYTE *)(v12 + 8) )
            {
              case 0:
              case 8:
              case 0xA:
              case 0xC:
              case 0x10:
                v20 = *(_QWORD *)(v12 + 32);
                v12 = *(_QWORD *)(v12 + 24);
                v13 *= v20;
                continue;
              case 1:
                v15 = 16;
                break;
              case 2:
                v15 = 32;
                break;
              case 3:
              case 9:
                v15 = 64;
                break;
              case 4:
                v15 = 80;
                break;
              case 5:
              case 6:
                v15 = 128;
                break;
              case 7:
                v27 = v13;
                v18 = sub_15A9520(a1, 0);
                v13 = v27;
                v15 = (unsigned int)(8 * v18);
                break;
              case 0xB:
                v15 = *(_DWORD *)(v12 + 8) >> 8;
                break;
              case 0xD:
                v26 = v13;
                v17 = (_QWORD *)sub_15A9930(a1, v12);
                v13 = v26;
                v15 = 8LL * *v17;
                break;
              case 0xE:
                v21 = v13;
                v22 = *(_QWORD *)(v12 + 24);
                v25 = *(_QWORD *)(v12 + 32);
                v23 = (unsigned int)sub_15A9FE0(a1, v22);
                v16 = sub_127FA20(a1, v22);
                v13 = v21;
                v15 = 8 * v25 * v23 * ((v23 + ((unsigned __int64)(v16 + 7) >> 3) - 1) / v23);
                break;
              case 0xF:
                v28 = v13;
                v19 = sub_15A9520(a1, *(_DWORD *)(v12 + 8) >> 8);
                v13 = v28;
                v15 = (unsigned int)(8 * v19);
                break;
            }
            break;
          }
          v8 = 8 * v14 * v10 * ((v14 + ((unsigned __int64)(v15 * v13 + 7) >> 3) - 1) / v14);
          break;
        case 0xF:
          v8 = 8 * (unsigned int)sub_15A9520(a1, *(_DWORD *)(v6 + 8) >> 8);
          break;
      }
      break;
    }
    result = a3 * (unsigned int)((unsigned __int64)(v8 * v7 + 7) >> 3);
    if ( (unsigned int)result > 0xF )
    {
      return 16;
    }
    else if ( !(_DWORD)result || ((unsigned int)result & ((_DWORD)result - 1)) != 0 )
    {
      return ((unsigned int)(((result >> 1) | result) >> 2) | (unsigned int)(result >> 1) | (unsigned int)result) + 1;
    }
  }
  return result;
}
