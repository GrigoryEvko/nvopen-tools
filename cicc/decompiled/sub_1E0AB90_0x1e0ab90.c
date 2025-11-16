// Function: sub_1E0AB90
// Address: 0x1e0ab90
//
__int64 __fastcall sub_1E0AB90(__int64 a1, __int64 a2)
{
  char v3; // al
  unsigned int v4; // r8d
  __int64 v6; // r12
  __int64 v7; // r14
  unsigned __int64 v8; // rbx
  __int64 v9; // rax
  unsigned __int64 v10; // rax
  __int64 v11; // r15
  unsigned int v12; // eax
  __int64 v13; // rsi
  __int64 v14; // rcx
  unsigned __int64 v15; // r14
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  _QWORD *v19; // rax
  int v20; // eax
  int v21; // eax
  __int64 v22; // rax
  __int64 v23; // [rsp+0h] [rbp-50h]
  __int64 v24; // [rsp+8h] [rbp-48h]
  unsigned __int64 v25; // [rsp+10h] [rbp-40h]
  __int64 v26; // [rsp+18h] [rbp-38h]
  __int64 v27; // [rsp+18h] [rbp-38h]
  __int64 v28; // [rsp+18h] [rbp-38h]
  __int64 v29; // [rsp+18h] [rbp-38h]
  __int64 v30; // [rsp+18h] [rbp-38h]

  v3 = sub_1E0AB70(a1);
  v4 = 18;
  if ( !v3 )
  {
    v6 = 1;
    v7 = sub_1E0AB50(a1);
    v8 = (unsigned int)sub_15A9FE0(a2, v7);
    while ( 2 )
    {
      switch ( *(_BYTE *)(v7 + 8) )
      {
        case 0:
        case 8:
        case 0xA:
        case 0xC:
        case 0x10:
          v16 = *(_QWORD *)(v7 + 32);
          v7 = *(_QWORD *)(v7 + 24);
          v6 *= v16;
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
          v9 = 8 * (unsigned int)sub_15A9520(a2, 0);
          break;
        case 0xB:
          v9 = *(_DWORD *)(v7 + 8) >> 8;
          break;
        case 0xD:
          v9 = 8LL * *(_QWORD *)sub_15A9930(a2, v7);
          break;
        case 0xE:
          v11 = *(_QWORD *)(v7 + 32);
          v26 = *(_QWORD *)(v7 + 24);
          v12 = sub_15A9FE0(a2, v26);
          v13 = v26;
          v14 = 1;
          v15 = v12;
          while ( 2 )
          {
            switch ( *(_BYTE *)(v13 + 8) )
            {
              case 0:
              case 8:
              case 0xA:
              case 0xC:
              case 0x10:
                v22 = *(_QWORD *)(v13 + 32);
                v13 = *(_QWORD *)(v13 + 24);
                v14 *= v22;
                continue;
              case 1:
                v17 = 16;
                break;
              case 2:
                v17 = 32;
                break;
              case 3:
              case 9:
                v17 = 64;
                break;
              case 4:
                v17 = 80;
                break;
              case 5:
              case 6:
                v17 = 128;
                break;
              case 7:
                v29 = v14;
                v20 = sub_15A9520(a2, 0);
                v14 = v29;
                v17 = (unsigned int)(8 * v20);
                break;
              case 0xB:
                v17 = *(_DWORD *)(v13 + 8) >> 8;
                break;
              case 0xD:
                v28 = v14;
                v19 = (_QWORD *)sub_15A9930(a2, v13);
                v14 = v28;
                v17 = 8LL * *v19;
                break;
              case 0xE:
                v23 = v14;
                v24 = *(_QWORD *)(v13 + 24);
                v27 = *(_QWORD *)(v13 + 32);
                v25 = (unsigned int)sub_15A9FE0(a2, v24);
                v18 = sub_127FA20(a2, v24);
                v14 = v23;
                v17 = 8 * v25 * v27 * ((v25 + ((unsigned __int64)(v18 + 7) >> 3) - 1) / v25);
                break;
              case 0xF:
                v30 = v14;
                v21 = sub_15A9520(a2, *(_DWORD *)(v13 + 8) >> 8);
                v14 = v30;
                v17 = (unsigned int)(8 * v21);
                break;
            }
            break;
          }
          v9 = 8 * v15 * v11 * ((v15 + ((unsigned __int64)(v17 * v14 + 7) >> 3) - 1) / v15);
          break;
        case 0xF:
          v9 = 8 * (unsigned int)sub_15A9520(a2, *(_DWORD *)(v7 + 8) >> 8);
          break;
      }
      break;
    }
    v10 = v8 * ((v8 + ((unsigned __int64)(v9 * v6 + 7) >> 3) - 1) / v8);
    if ( v10 == 16 )
      return 9;
    if ( v10 > 0x10 )
    {
      v4 = 10;
      if ( v10 == 32 )
        return v4;
    }
    else
    {
      v4 = 7;
      if ( v10 == 4 )
        return v4;
      if ( v10 == 8 )
      {
        LOBYTE(v4) = 8;
        return v4;
      }
    }
    LOBYTE(v4) = 3;
  }
  return v4;
}
