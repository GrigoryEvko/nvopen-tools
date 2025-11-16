// Function: sub_1B6F4A0
// Address: 0x1b6f4a0
//
__int64 __fastcall sub_1B6F4A0(__int64 a1, unsigned __int16 *a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rsi
  __int64 v7; // rax
  unsigned __int16 *v8; // r15
  __int64 v9; // r14
  __int64 v10; // rax
  __int64 v12; // rax
  unsigned int v13; // eax
  __int64 v14; // rsi
  __int64 v15; // r10
  unsigned __int64 v16; // r8
  __int64 v17; // rax
  __int64 v18; // rax
  unsigned int v19; // esi
  int v20; // eax
  __int64 v21; // rax
  _QWORD *v22; // rax
  __int64 v23; // [rsp+0h] [rbp-60h]
  unsigned __int64 v24; // [rsp+8h] [rbp-58h]
  __int64 v25; // [rsp+10h] [rbp-50h]
  __int64 v26; // [rsp+18h] [rbp-48h]
  unsigned __int64 v27; // [rsp+18h] [rbp-48h]
  __int64 v28; // [rsp+18h] [rbp-48h]
  __int64 v29; // [rsp+20h] [rbp-40h]
  unsigned __int64 v30; // [rsp+20h] [rbp-40h]
  __int64 v31; // [rsp+20h] [rbp-40h]
  unsigned __int64 v32; // [rsp+20h] [rbp-40h]
  __int64 v33; // [rsp+28h] [rbp-38h]

  v6 = **(_QWORD **)(a3 - 48);
  v7 = *(unsigned __int8 *)(v6 + 8);
  if ( (unsigned __int8)(v7 - 13) <= 1u )
    return 0xFFFFFFFFLL;
  v8 = *(unsigned __int16 **)(a3 - 24);
  v9 = 1;
  while ( 2 )
  {
    switch ( v7 )
    {
      case 0LL:
      case 8LL:
      case 10LL:
      case 12LL:
      case 16LL:
        v12 = *(_QWORD *)(v6 + 32);
        v6 = *(_QWORD *)(v6 + 24);
        v9 *= v12;
        v7 = *(unsigned __int8 *)(v6 + 8);
        continue;
      case 1LL:
        v10 = 16;
        break;
      case 2LL:
        v10 = 32;
        break;
      case 3LL:
      case 9LL:
        v10 = 64;
        break;
      case 4LL:
        v10 = 80;
        break;
      case 5LL:
      case 6LL:
        v10 = 128;
        break;
      case 7LL:
        v10 = 8 * (unsigned int)sub_15A9520(a4, 0);
        break;
      case 11LL:
        v10 = *(_DWORD *)(v6 + 8) >> 8;
        break;
      case 13LL:
        v10 = 8LL * *(_QWORD *)sub_15A9930(a4, v6);
        break;
      case 14LL:
        v29 = *(_QWORD *)(v6 + 24);
        v33 = *(_QWORD *)(v6 + 32);
        v13 = sub_15A9FE0(a4, v29);
        v14 = v29;
        v15 = 1;
        v16 = v13;
        while ( 2 )
        {
          switch ( *(_BYTE *)(v14 + 8) )
          {
            case 0:
            case 8:
            case 0xA:
            case 0xC:
            case 0x10:
              v18 = *(_QWORD *)(v14 + 32);
              v14 = *(_QWORD *)(v14 + 24);
              v15 *= v18;
              continue;
            case 1:
              v17 = 16;
              goto LABEL_18;
            case 2:
              v17 = 32;
              goto LABEL_18;
            case 3:
            case 9:
              v17 = 64;
              goto LABEL_18;
            case 4:
              v17 = 80;
              goto LABEL_18;
            case 5:
            case 6:
              v17 = 128;
              goto LABEL_18;
            case 7:
              v26 = v15;
              v19 = 0;
              v30 = v16;
              goto LABEL_25;
            case 0xB:
              v17 = *(_DWORD *)(v14 + 8) >> 8;
              goto LABEL_18;
            case 0xD:
              v28 = v15;
              v32 = v16;
              v22 = (_QWORD *)sub_15A9930(a4, v14);
              v16 = v32;
              v15 = v28;
              v17 = 8LL * *v22;
              goto LABEL_18;
            case 0xE:
              v23 = v15;
              v24 = v16;
              v25 = *(_QWORD *)(v14 + 24);
              v31 = *(_QWORD *)(v14 + 32);
              v27 = (unsigned int)sub_15A9FE0(a4, v25);
              v21 = sub_127FA20(a4, v25);
              v16 = v24;
              v15 = v23;
              v17 = 8 * v31 * v27 * ((v27 + ((unsigned __int64)(v21 + 7) >> 3) - 1) / v27);
              goto LABEL_18;
            case 0xF:
              v26 = v15;
              v30 = v16;
              v19 = *(_DWORD *)(v14 + 8) >> 8;
LABEL_25:
              v20 = sub_15A9520(a4, v19);
              v16 = v30;
              v15 = v26;
              v17 = (unsigned int)(8 * v20);
LABEL_18:
              v10 = 8 * v16 * v33 * ((v16 + ((unsigned __int64)(v17 * v15 + 7) >> 3) - 1) / v16);
              break;
          }
          break;
        }
        break;
      case 15LL:
        v10 = 8 * (unsigned int)sub_15A9520(a4, *(_DWORD *)(v6 + 8) >> 8);
        break;
    }
    return sub_1B6E190(a1, a2, v8, v10 * v9, a4);
  }
}
