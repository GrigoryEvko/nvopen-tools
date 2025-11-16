// Function: sub_3976960
// Address: 0x3976960
//
void __fastcall sub_3976960(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v5; // rbx
  __int64 v6; // r15
  unsigned __int64 v7; // rcx
  __int64 v8; // rax
  _QWORD *v9; // rax
  unsigned int v10; // eax
  __int64 v11; // rsi
  __int64 v12; // rdx
  unsigned __int64 v13; // r15
  int v14; // eax
  __int64 v15; // rax
  int v16; // eax
  __int64 v17; // rax
  unsigned int v18; // esi
  int v19; // eax
  __int64 v20; // rax
  unsigned __int64 v21; // rax
  _QWORD *v22; // rax
  __int64 v23; // [rsp+0h] [rbp-50h]
  unsigned __int64 v24; // [rsp+8h] [rbp-48h]
  __int64 v25; // [rsp+8h] [rbp-48h]
  __int64 v26; // [rsp+8h] [rbp-48h]
  __int64 v27; // [rsp+10h] [rbp-40h]
  unsigned __int64 v28; // [rsp+10h] [rbp-40h]
  __int64 v29; // [rsp+10h] [rbp-40h]
  unsigned __int64 v30; // [rsp+10h] [rbp-40h]
  unsigned __int64 v31; // [rsp+18h] [rbp-38h]
  __int64 v32; // [rsp+18h] [rbp-38h]
  unsigned __int64 v33; // [rsp+18h] [rbp-38h]
  unsigned __int64 v34; // [rsp+18h] [rbp-38h]

  v5 = 1;
  v6 = *a3;
  v7 = (unsigned int)sub_15A9FE0(a2, *a3);
  while ( 2 )
  {
    switch ( *(_BYTE *)(v6 + 8) )
    {
      case 0:
      case 8:
      case 0xA:
      case 0xC:
      case 0x10:
        v15 = *(_QWORD *)(v6 + 32);
        v6 = *(_QWORD *)(v6 + 24);
        v5 *= v15;
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
        v34 = v7;
        v16 = sub_15A9520(a2, 0);
        v7 = v34;
        v8 = (unsigned int)(8 * v16);
        break;
      case 0xB:
        v8 = *(_DWORD *)(v6 + 8) >> 8;
        break;
      case 0xD:
        v31 = v7;
        v9 = (_QWORD *)sub_15A9930(a2, v6);
        v7 = v31;
        v8 = 8LL * *v9;
        break;
      case 0xE:
        v24 = v7;
        v27 = *(_QWORD *)(v6 + 24);
        v32 = *(_QWORD *)(v6 + 32);
        v10 = sub_15A9FE0(a2, v27);
        v7 = v24;
        v11 = v27;
        v12 = 1;
        v13 = v10;
        while ( 2 )
        {
          switch ( *(_BYTE *)(v11 + 8) )
          {
            case 0:
            case 8:
            case 0xA:
            case 0xC:
            case 0x10:
              v20 = *(_QWORD *)(v11 + 32);
              v11 = *(_QWORD *)(v11 + 24);
              v12 *= v20;
              continue;
            case 1:
              v17 = 16;
              goto LABEL_21;
            case 2:
              v17 = 32;
              goto LABEL_21;
            case 3:
            case 9:
              v17 = 64;
              goto LABEL_21;
            case 4:
              v17 = 80;
              goto LABEL_21;
            case 5:
            case 6:
              v17 = 128;
              goto LABEL_21;
            case 7:
              v25 = v12;
              v18 = 0;
              v28 = v7;
              goto LABEL_27;
            case 0xB:
              v17 = *(_DWORD *)(v11 + 8) >> 8;
              goto LABEL_21;
            case 0xD:
              v26 = v12;
              v30 = v7;
              v22 = (_QWORD *)sub_15A9930(a2, v11);
              v7 = v30;
              v12 = v26;
              v17 = 8LL * *v22;
              goto LABEL_21;
            case 0xE:
              v23 = v12;
              v29 = *(_QWORD *)(v11 + 32);
              v21 = sub_12BE0A0(a2, *(_QWORD *)(v11 + 24));
              v7 = v24;
              v12 = v23;
              v17 = 8 * v29 * v21;
              goto LABEL_21;
            case 0xF:
              v25 = v12;
              v28 = v7;
              v18 = *(_DWORD *)(v11 + 8) >> 8;
LABEL_27:
              v19 = sub_15A9520(a2, v18);
              v7 = v28;
              v12 = v25;
              v17 = (unsigned int)(8 * v19);
LABEL_21:
              v8 = 8 * v13 * v32 * ((v13 + ((unsigned __int64)(v17 * v12 + 7) >> 3) - 1) / v13);
              break;
          }
          break;
        }
        break;
      case 0xF:
        v33 = v7;
        v14 = sub_15A9520(a2, *(_DWORD *)(v6 + 8) >> 8);
        v7 = v33;
        v8 = (unsigned int)(8 * v14);
        break;
    }
    break;
  }
  if ( v7 * ((v7 + ((unsigned __int64)(v8 * v5 + 7) >> 3) - 1) / v7) )
  {
    sub_39740F0(a2, (__int64)a3, a1, 0, 0);
  }
  else if ( *(_BYTE *)(*(_QWORD *)(a1 + 240) + 18LL) )
  {
    (*(void (__fastcall **)(_QWORD, _QWORD, __int64))(**(_QWORD **)(a1 + 256) + 424LL))(*(_QWORD *)(a1 + 256), 0, 1);
  }
}
