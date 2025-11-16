// Function: sub_182C650
// Address: 0x182c650
//
unsigned __int64 __fastcall sub_182C650(__int64 a1)
{
  __int64 v1; // rdx
  __int64 v2; // r12
  __int64 v3; // r15
  __int64 v4; // rbx
  __int64 v5; // rax
  __int64 v6; // r14
  unsigned __int64 v7; // r13
  __int64 v8; // rax
  __int64 v10; // rax
  unsigned int v11; // eax
  __int64 v12; // rsi
  __int64 v13; // rcx
  unsigned __int64 v14; // r15
  __int64 v15; // rax
  __int64 v16; // rax
  int v17; // eax
  int v18; // eax
  __int64 v19; // rax
  _QWORD *v20; // rax
  __int64 v21; // [rsp+8h] [rbp-58h]
  __int64 v22; // [rsp+10h] [rbp-50h]
  unsigned __int64 v23; // [rsp+18h] [rbp-48h]
  __int64 v24; // [rsp+20h] [rbp-40h]
  __int64 v25; // [rsp+20h] [rbp-40h]
  __int64 v26; // [rsp+20h] [rbp-40h]
  __int64 v27; // [rsp+20h] [rbp-40h]
  __int64 v28; // [rsp+20h] [rbp-40h]
  __int64 v29; // [rsp+28h] [rbp-38h]

  if ( (unsigned __int8)sub_15F8BF0(a1) )
  {
    v1 = *(_QWORD *)(a1 - 24);
    if ( *(_BYTE *)(v1 + 16) != 13 )
      BUG();
    v2 = *(_QWORD *)(v1 + 24);
    if ( *(_DWORD *)(v1 + 32) > 0x40u )
      v2 = *(_QWORD *)v2;
  }
  else
  {
    v2 = 1;
  }
  v3 = *(_QWORD *)(a1 + 56);
  v4 = 1;
  v5 = sub_15F2050(a1);
  v6 = sub_1632FA0(v5);
  v7 = (unsigned int)sub_15A9FE0(v6, v3);
  while ( 2 )
  {
    switch ( *(_BYTE *)(v3 + 8) )
    {
      case 0:
      case 8:
      case 0xA:
      case 0xC:
      case 0x10:
        v10 = *(_QWORD *)(v3 + 32);
        v3 = *(_QWORD *)(v3 + 24);
        v4 *= v10;
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
        v8 = 8 * (unsigned int)sub_15A9520(v6, 0);
        break;
      case 0xB:
        v8 = *(_DWORD *)(v3 + 8) >> 8;
        break;
      case 0xD:
        v8 = 8LL * *(_QWORD *)sub_15A9930(v6, v3);
        break;
      case 0xE:
        v24 = *(_QWORD *)(v3 + 24);
        v29 = *(_QWORD *)(v3 + 32);
        v11 = sub_15A9FE0(v6, v24);
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
              v16 = *(_QWORD *)(v12 + 32);
              v12 = *(_QWORD *)(v12 + 24);
              v13 *= v16;
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
              v25 = v13;
              v17 = sub_15A9520(v6, 0);
              v13 = v25;
              v15 = (unsigned int)(8 * v17);
              break;
            case 0xB:
              v15 = *(_DWORD *)(v12 + 8) >> 8;
              break;
            case 0xD:
              v28 = v13;
              v20 = (_QWORD *)sub_15A9930(v6, v12);
              v13 = v28;
              v15 = 8LL * *v20;
              break;
            case 0xE:
              v21 = v13;
              v22 = *(_QWORD *)(v12 + 24);
              v27 = *(_QWORD *)(v12 + 32);
              v23 = (unsigned int)sub_15A9FE0(v6, v22);
              v19 = sub_127FA20(v6, v22);
              v13 = v21;
              v15 = 8 * v27 * v23 * ((v23 + ((unsigned __int64)(v19 + 7) >> 3) - 1) / v23);
              break;
            case 0xF:
              v26 = v13;
              v18 = sub_15A9520(v6, *(_DWORD *)(v12 + 8) >> 8);
              v13 = v26;
              v15 = (unsigned int)(8 * v18);
              break;
          }
          break;
        }
        v8 = 8 * v14 * v29 * ((v14 + ((unsigned __int64)(v15 * v13 + 7) >> 3) - 1) / v14);
        break;
      case 0xF:
        v8 = 8 * (unsigned int)sub_15A9520(v6, *(_DWORD *)(v3 + 8) >> 8);
        break;
    }
    return v7 * v2 * ((v7 + ((unsigned __int64)(v8 * v4 + 7) >> 3) - 1) / v7);
  }
}
