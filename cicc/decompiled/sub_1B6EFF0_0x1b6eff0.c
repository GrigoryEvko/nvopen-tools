// Function: sub_1B6EFF0
// Address: 0x1b6eff0
//
bool __fastcall sub_1B6EFF0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rsi
  __int64 v5; // rax
  bool result; // al
  __int64 v9; // r8
  char v10; // r14
  __int64 v11; // r15
  unsigned __int64 v12; // r15
  __int64 v13; // rsi
  __int64 v14; // r8
  _QWORD *v15; // rax
  unsigned int v16; // eax
  __int64 v17; // rsi
  __int64 v18; // r10
  unsigned __int64 v19; // r15
  int v20; // eax
  __int64 v21; // rax
  int v22; // eax
  __int64 v23; // rax
  char v24; // r8
  _DWORD *v25; // rdi
  _DWORD *v26; // rsi
  _DWORD *v27; // rdi
  _DWORD *v28; // rsi
  __int64 v29; // rax
  int v30; // eax
  __int64 v31; // rax
  _QWORD *v32; // rax
  int v33; // eax
  __int64 v34; // rax
  __int64 v35; // rax
  unsigned int v36; // esi
  int v37; // eax
  __int64 v38; // rax
  _QWORD *v39; // rax
  __int64 v40; // [rsp+0h] [rbp-70h]
  __int64 v41; // [rsp+8h] [rbp-68h]
  __int64 v42; // [rsp+10h] [rbp-60h]
  __int64 v43; // [rsp+10h] [rbp-60h]
  __int64 v44; // [rsp+18h] [rbp-58h]
  __int64 v45; // [rsp+18h] [rbp-58h]
  __int64 v46; // [rsp+18h] [rbp-58h]
  unsigned __int64 v47; // [rsp+18h] [rbp-58h]
  __int64 v48; // [rsp+18h] [rbp-58h]
  __int64 v49; // [rsp+20h] [rbp-50h]
  unsigned __int64 v50; // [rsp+20h] [rbp-50h]
  __int64 v51; // [rsp+20h] [rbp-50h]
  __int64 v52; // [rsp+20h] [rbp-50h]
  __int64 v53; // [rsp+20h] [rbp-50h]
  __int64 v54; // [rsp+28h] [rbp-48h]
  __int64 v55; // [rsp+28h] [rbp-48h]
  __int64 v56; // [rsp+28h] [rbp-48h]
  __int64 v57; // [rsp+28h] [rbp-48h]
  __int64 v58; // [rsp+28h] [rbp-48h]
  __int64 v59; // [rsp+28h] [rbp-48h]
  __int64 v60; // [rsp+28h] [rbp-48h]
  __int64 v61; // [rsp+28h] [rbp-48h]
  int v62[13]; // [rsp+3Ch] [rbp-34h] BYREF

  v4 = *a1;
  v5 = *(unsigned __int8 *)(*a1 + 8);
  if ( (_BYTE)v5 == 13 )
    return **(_QWORD **)(v4 + 16) == a2;
  v9 = 1;
  v10 = (unsigned __int8)(v5 - 13) <= 1u || (unsigned __int8)(*(_BYTE *)(a2 + 8) - 13) <= 1u;
  if ( v10 )
    return 0;
  while ( 2 )
  {
    switch ( v5 )
    {
      case 0LL:
      case 8LL:
      case 10LL:
      case 12LL:
      case 16LL:
        v21 = *(_QWORD *)(v4 + 32);
        v4 = *(_QWORD *)(v4 + 24);
        v9 *= v21;
        v5 = *(unsigned __int8 *)(v4 + 8);
        continue;
      case 1LL:
        v11 = 16;
        break;
      case 2LL:
        v11 = 32;
        break;
      case 3LL:
      case 9LL:
        v11 = 64;
        break;
      case 4LL:
        v11 = 80;
        break;
      case 5LL:
      case 6LL:
        v11 = 128;
        break;
      case 7LL:
        v57 = v9;
        v22 = sub_15A9520(a3, 0);
        v9 = v57;
        v11 = (unsigned int)(8 * v22);
        break;
      case 11LL:
        v11 = *(_DWORD *)(v4 + 8) >> 8;
        break;
      case 13LL:
        v54 = v9;
        v15 = (_QWORD *)sub_15A9930(a3, v4);
        v9 = v54;
        v11 = 8LL * *v15;
        break;
      case 14LL:
        v44 = v9;
        v49 = *(_QWORD *)(v4 + 24);
        v55 = *(_QWORD *)(v4 + 32);
        v16 = sub_15A9FE0(a3, v49);
        v9 = v44;
        v17 = v49;
        v18 = 1;
        v19 = v16;
        while ( 2 )
        {
          switch ( *(_BYTE *)(v17 + 8) )
          {
            case 0:
            case 8:
            case 0xA:
            case 0xC:
            case 0x10:
              v35 = *(_QWORD *)(v17 + 32);
              v17 = *(_QWORD *)(v17 + 24);
              v18 *= v35;
              continue;
            case 1:
              v34 = 16;
              goto LABEL_40;
            case 2:
              v34 = 32;
              goto LABEL_40;
            case 3:
            case 9:
              v34 = 64;
              goto LABEL_40;
            case 4:
              v34 = 80;
              goto LABEL_40;
            case 5:
            case 6:
              v34 = 128;
              goto LABEL_40;
            case 7:
              v46 = v18;
              v36 = 0;
              v51 = v9;
              goto LABEL_47;
            case 0xB:
              v34 = *(_DWORD *)(v17 + 8) >> 8;
              goto LABEL_40;
            case 0xD:
              v48 = v18;
              v53 = v9;
              v39 = (_QWORD *)sub_15A9930(a3, v17);
              v9 = v53;
              v18 = v48;
              v34 = 8LL * *v39;
              goto LABEL_40;
            case 0xE:
              v40 = v18;
              v41 = v44;
              v43 = *(_QWORD *)(v17 + 24);
              v52 = *(_QWORD *)(v17 + 32);
              v47 = (unsigned int)sub_15A9FE0(a3, v43);
              v38 = sub_127FA20(a3, v43);
              v9 = v41;
              v18 = v40;
              v34 = 8 * v52 * v47 * ((v47 + ((unsigned __int64)(v38 + 7) >> 3) - 1) / v47);
              goto LABEL_40;
            case 0xF:
              v46 = v18;
              v51 = v9;
              v36 = *(_DWORD *)(v17 + 8) >> 8;
LABEL_47:
              v37 = sub_15A9520(a3, v36);
              v9 = v51;
              v18 = v46;
              v34 = (unsigned int)(8 * v37);
LABEL_40:
              v11 = 8 * v55 * v19 * ((v19 + ((unsigned __int64)(v34 * v18 + 7) >> 3) - 1) / v19);
              break;
          }
          break;
        }
        break;
      case 15LL:
        v56 = v9;
        v20 = sub_15A9520(a3, *(_DWORD *)(v4 + 8) >> 8);
        v9 = v56;
        v11 = (unsigned int)(8 * v20);
        break;
    }
    break;
  }
  v12 = v9 * v11;
  if ( ((v12 + 7) & 0xFFFFFFFFFFFFFFF8LL) != v12 )
    return 0;
  v13 = a2;
  v14 = 1;
  while ( 2 )
  {
    switch ( *(_BYTE *)(v13 + 8) )
    {
      case 1:
        v23 = 16;
        goto LABEL_23;
      case 2:
        v23 = 32;
        goto LABEL_23;
      case 3:
      case 9:
        v23 = 64;
        goto LABEL_23;
      case 4:
        v23 = 80;
        goto LABEL_23;
      case 5:
      case 6:
        v23 = 128;
        goto LABEL_23;
      case 7:
        v61 = v14;
        v33 = sub_15A9520(a3, 0);
        v14 = v61;
        v23 = (unsigned int)(8 * v33);
        goto LABEL_23;
      case 0xB:
        v23 = *(_DWORD *)(v13 + 8) >> 8;
        goto LABEL_23;
      case 0xD:
        v60 = v14;
        v32 = (_QWORD *)sub_15A9930(a3, v13);
        v14 = v60;
        v23 = 8LL * *v32;
        goto LABEL_23;
      case 0xE:
        v42 = v14;
        v45 = *(_QWORD *)(v13 + 24);
        v59 = *(_QWORD *)(v13 + 32);
        v50 = (unsigned int)sub_15A9FE0(a3, v45);
        v31 = sub_127FA20(a3, v45);
        v14 = v42;
        v23 = 8 * v50 * v59 * ((v50 + ((unsigned __int64)(v31 + 7) >> 3) - 1) / v50);
        goto LABEL_23;
      case 0xF:
        v58 = v14;
        v30 = sub_15A9520(a3, *(_DWORD *)(v13 + 8) >> 8);
        v14 = v58;
        v23 = (unsigned int)(8 * v30);
LABEL_23:
        if ( v12 < v23 * v14 )
          return 0;
        v24 = 0;
        if ( *(_BYTE *)(*a1 + 8) == 15 )
        {
          v25 = *(_DWORD **)(a3 + 408);
          v26 = &v25[*(unsigned int *)(a3 + 416)];
          v62[0] = *(_DWORD *)(*a1 + 8) >> 8;
          v24 = v26 != sub_1B6E0D0(v25, (__int64)v26, v62);
        }
        if ( *(_BYTE *)(a2 + 8) == 15 )
        {
          v27 = *(_DWORD **)(a3 + 408);
          v28 = &v27[*(unsigned int *)(a3 + 416)];
          v62[0] = *(_DWORD *)(a2 + 8) >> 8;
          v10 = v28 != sub_1B6E0D0(v27, (__int64)v28, v62);
        }
        result = v10 == v24;
        break;
      case 0x10:
        v29 = *(_QWORD *)(v13 + 32);
        v13 = *(_QWORD *)(v13 + 24);
        v14 *= v29;
        continue;
      default:
        BUG();
    }
    return result;
  }
}
