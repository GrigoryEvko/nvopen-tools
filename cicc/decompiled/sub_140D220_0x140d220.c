// Function: sub_140D220
// Address: 0x140d220
//
_QWORD *__fastcall sub_140D220(__int64 *a1, __int64 a2)
{
  __int64 v3; // rbx
  __int64 v4; // r14
  unsigned __int64 v5; // rax
  __int64 v7; // rdx
  _QWORD *v8; // r13
  __int64 v9; // rdi
  unsigned __int64 v10; // r15
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  unsigned int v14; // eax
  __int64 v15; // rsi
  __int64 v16; // rdi
  __int64 v17; // rdx
  unsigned __int64 v18; // r14
  __int64 v19; // rax
  int v20; // eax
  int v21; // eax
  unsigned __int64 v22; // rax
  _QWORD *v23; // rax
  __int64 v24; // rax
  __int64 v25; // [rsp+8h] [rbp-68h]
  __int64 v26; // [rsp+8h] [rbp-68h]
  __int64 v27; // [rsp+10h] [rbp-60h]
  __int64 v28; // [rsp+10h] [rbp-60h]
  __int64 v29; // [rsp+10h] [rbp-60h]
  __int64 v30; // [rsp+10h] [rbp-60h]
  __int64 v31; // [rsp+10h] [rbp-60h]
  __int64 v32; // [rsp+18h] [rbp-58h]
  __int64 v33; // [rsp+18h] [rbp-58h]
  char v34[16]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v35; // [rsp+30h] [rbp-40h]

  v3 = a2;
  v4 = *(_QWORD *)(a2 + 56);
  v5 = *(unsigned __int8 *)(v4 + 8);
  if ( (unsigned __int8)v5 > 0xFu || (v7 = 35454, !_bittest64(&v7, v5)) )
  {
LABEL_2:
    if ( (unsigned int)(v5 - 13) > 1 && (_DWORD)v5 != 16 || !(unsigned __int8)sub_16435F0(v4, 0) )
      return 0;
    v4 = *(_QWORD *)(v3 + 56);
  }
  v8 = *(_QWORD **)(v3 - 24);
  v3 = 1;
  v32 = *a1;
  v9 = *a1;
  v10 = (unsigned int)sub_15A9FE0(*a1, v4);
  while ( 2 )
  {
    LODWORD(v5) = *(unsigned __int8 *)(v4 + 8);
    switch ( (char)v5 )
    {
      case 0:
      case 8:
      case 10:
      case 12:
        v13 = *(_QWORD *)(v4 + 32);
        v4 = *(_QWORD *)(v4 + 24);
        v3 *= v13;
        continue;
      case 1:
        v11 = 16;
        break;
      case 2:
        v11 = 32;
        break;
      case 3:
      case 9:
        v11 = 64;
        break;
      case 4:
        v11 = 80;
        break;
      case 5:
      case 6:
        v11 = 128;
        break;
      case 7:
        v11 = 8 * (unsigned int)sub_15A9520(v9, 0);
        break;
      case 11:
        v11 = *(_DWORD *)(v4 + 8) >> 8;
        break;
      case 13:
        v11 = 8LL * *(_QWORD *)sub_15A9930(v9, v4);
        break;
      case 14:
        v27 = v32;
        v25 = *(_QWORD *)(v4 + 24);
        v33 = *(_QWORD *)(v4 + 32);
        v14 = sub_15A9FE0(v9, v25);
        v15 = v25;
        v16 = v27;
        v17 = 1;
        v18 = v14;
        while ( 2 )
        {
          switch ( *(_BYTE *)(v15 + 8) )
          {
            case 0:
            case 8:
            case 0xA:
            case 0xC:
            case 0x10:
              v24 = *(_QWORD *)(v15 + 32);
              v15 = *(_QWORD *)(v15 + 24);
              v17 *= v24;
              continue;
            case 1:
              v19 = 16;
              break;
            case 2:
              v19 = 32;
              break;
            case 3:
            case 9:
              v19 = 64;
              break;
            case 4:
              v19 = 80;
              break;
            case 5:
            case 6:
              v19 = 128;
              break;
            case 7:
              v28 = v17;
              v20 = sub_15A9520(v16, 0);
              v17 = v28;
              v19 = (unsigned int)(8 * v20);
              break;
            case 0xB:
              v19 = *(_DWORD *)(v15 + 8) >> 8;
              break;
            case 0xD:
              v31 = v17;
              v23 = (_QWORD *)sub_15A9930(v16, v15);
              v17 = v31;
              v19 = 8LL * *v23;
              break;
            case 0xE:
              v26 = v17;
              v30 = *(_QWORD *)(v15 + 32);
              v22 = sub_12BE0A0(v16, *(_QWORD *)(v15 + 24));
              v17 = v26;
              v19 = 8 * v30 * v22;
              break;
            case 0xF:
              v29 = v17;
              v21 = sub_15A9520(v16, *(_DWORD *)(v15 + 8) >> 8);
              v17 = v29;
              v19 = (unsigned int)(8 * v21);
              break;
          }
          break;
        }
        v11 = 8 * v18 * v33 * ((v18 + ((unsigned __int64)(v19 * v17 + 7) >> 3) - 1) / v18);
        break;
      case 15:
        v11 = 8 * (unsigned int)sub_15A9520(v9, *(_DWORD *)(v4 + 8) >> 8);
        break;
      default:
        goto LABEL_2;
    }
    break;
  }
  v12 = sub_15A0680(*v8, v10 * ((v10 + ((unsigned __int64)(v11 * v3 + 7) >> 3) - 1) / v10), 0);
  v35 = 257;
  return sub_140D0B0(a1 + 3, v12, (__int64)v8, (__int64)v34, 0, 0);
}
