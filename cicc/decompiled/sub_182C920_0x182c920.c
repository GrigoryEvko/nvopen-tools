// Function: sub_182C920
// Address: 0x182c920
//
__int64 __fastcall sub_182C920(__int64 a1, _BYTE *a2, unsigned __int64 *a3, int *a4)
{
  __int64 v7; // rax
  __int64 v8; // rdi
  char v9; // al
  __int64 v10; // rsi
  __int64 v11; // r15
  __int64 v12; // rax
  __int64 v13; // r12
  __int64 v14; // rax
  __int64 v16; // r15
  __int64 v17; // rsi
  __int64 v18; // r15
  __int64 v19; // rsi
  __int64 v20; // r15
  __int64 v21; // rsi
  __int64 v22; // r14
  unsigned int v23; // eax
  __int64 v24; // rsi
  __int64 v25; // r9
  unsigned __int64 v26; // r8
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rax
  unsigned __int64 v30; // r14
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // rax
  unsigned __int64 v34; // r14
  __int64 v35; // rax
  __int64 v36; // rax
  _QWORD *v37; // rax
  int v38; // eax
  __int64 v39; // rax
  int v40; // eax
  unsigned __int64 v41; // r14
  __int64 v42; // rax
  __int64 v43; // [rsp+0h] [rbp-60h]
  unsigned __int64 v44; // [rsp+8h] [rbp-58h]
  __int64 v45; // [rsp+10h] [rbp-50h]
  __int64 v46; // [rsp+18h] [rbp-48h]
  __int64 v47; // [rsp+18h] [rbp-48h]
  __int64 v48; // [rsp+18h] [rbp-48h]
  __int64 v49; // [rsp+20h] [rbp-40h]
  unsigned __int64 v50; // [rsp+20h] [rbp-40h]
  __int64 v51; // [rsp+20h] [rbp-40h]
  __int64 v52; // [rsp+20h] [rbp-40h]
  __int64 v53; // [rsp+20h] [rbp-40h]
  __int64 v54; // [rsp+28h] [rbp-38h]
  __int64 v55; // [rsp+28h] [rbp-38h]
  __int64 v56; // [rsp+28h] [rbp-38h]
  unsigned __int64 v57; // [rsp+28h] [rbp-38h]
  unsigned __int64 v58; // [rsp+28h] [rbp-38h]
  unsigned __int64 v59; // [rsp+28h] [rbp-38h]
  __int64 v60; // [rsp+28h] [rbp-38h]

  v7 = sub_15F2050(a1);
  v8 = sub_1632FA0(v7);
  v9 = *(_BYTE *)(a1 + 16);
  if ( v9 != 54 )
  {
    if ( v9 == 55 )
    {
      if ( byte_4FAA180 )
      {
        *a2 = 1;
        v16 = 1;
        v17 = **(_QWORD **)(a1 - 48);
        while ( 2 )
        {
          switch ( *(_BYTE *)(v17 + 8) )
          {
            case 1:
              v28 = 16;
              goto LABEL_39;
            case 2:
              v28 = 32;
              goto LABEL_39;
            case 3:
            case 9:
              v28 = 64;
              goto LABEL_39;
            case 4:
              v28 = 80;
              goto LABEL_39;
            case 5:
            case 6:
              v28 = 128;
              goto LABEL_39;
            case 7:
              v28 = 8 * (unsigned int)sub_15A9520(v8, 0);
              goto LABEL_39;
            case 0xB:
              v28 = *(_DWORD *)(v17 + 8) >> 8;
              goto LABEL_39;
            case 0xD:
              v28 = 8LL * *(_QWORD *)sub_15A9930(v8, v17);
              goto LABEL_39;
            case 0xE:
              v46 = *(_QWORD *)(v17 + 24);
              v54 = *(_QWORD *)(v17 + 32);
              v30 = (unsigned int)sub_15A9FE0(v8, v46);
              v28 = 8 * v54 * v30 * ((v30 + ((unsigned __int64)(sub_127FA20(v8, v46) + 7) >> 3) - 1) / v30);
              goto LABEL_39;
            case 0xF:
              v28 = 8 * (unsigned int)sub_15A9520(v8, *(_DWORD *)(v17 + 8) >> 8);
LABEL_39:
              *a3 = (v28 * v16 + 7) & 0xFFFFFFFFFFFFFFF8LL;
              goto LABEL_7;
            case 0x10:
              v29 = *(_QWORD *)(v17 + 32);
              v17 = *(_QWORD *)(v17 + 24);
              v16 *= v29;
              continue;
            default:
              goto LABEL_88;
          }
        }
      }
    }
    else if ( v9 == 59 )
    {
      if ( byte_4FAA0A0 )
      {
        *a2 = 1;
        v18 = 1;
        v19 = **(_QWORD **)(a1 - 24);
        while ( 2 )
        {
          switch ( *(_BYTE *)(v19 + 8) )
          {
            case 1:
              v31 = 16;
              goto LABEL_51;
            case 2:
              v31 = 32;
              goto LABEL_51;
            case 3:
            case 9:
              v31 = 64;
              goto LABEL_51;
            case 4:
              v31 = 80;
              goto LABEL_51;
            case 5:
            case 6:
              v31 = 128;
              goto LABEL_51;
            case 7:
              v31 = 8 * (unsigned int)sub_15A9520(v8, 0);
              goto LABEL_51;
            case 0xB:
              v31 = *(_DWORD *)(v19 + 8) >> 8;
              goto LABEL_51;
            case 0xD:
              v31 = 8LL * *(_QWORD *)sub_15A9930(v8, v19);
              goto LABEL_51;
            case 0xE:
              v47 = *(_QWORD *)(v19 + 24);
              v55 = *(_QWORD *)(v19 + 32);
              v34 = (unsigned int)sub_15A9FE0(v8, v47);
              v31 = 8 * v55 * v34 * ((v34 + ((unsigned __int64)(sub_127FA20(v8, v47) + 7) >> 3) - 1) / v34);
              goto LABEL_51;
            case 0xF:
              v31 = 8 * (unsigned int)sub_15A9520(v8, *(_DWORD *)(v19 + 8) >> 8);
LABEL_51:
              *a3 = (v31 * v18 + 7) & 0xFFFFFFFFFFFFFFF8LL;
              *a4 = 0;
              v13 = *(_QWORD *)(a1 - 48);
              if ( !v13 )
                return 0;
              goto LABEL_8;
            case 0x10:
              v33 = *(_QWORD *)(v19 + 32);
              v19 = *(_QWORD *)(v19 + 24);
              v18 *= v33;
              continue;
            default:
              goto LABEL_88;
          }
        }
      }
    }
    else if ( v9 == 58 && byte_4FAA0A0 )
    {
      *a2 = 1;
      v20 = 1;
      v21 = **(_QWORD **)(a1 - 48);
      while ( 2 )
      {
        switch ( *(_BYTE *)(v21 + 8) )
        {
          case 1:
            v35 = 16;
            goto LABEL_67;
          case 2:
            v35 = 32;
            goto LABEL_67;
          case 3:
          case 9:
            v35 = 64;
            goto LABEL_67;
          case 4:
            v35 = 80;
            goto LABEL_67;
          case 5:
          case 6:
            v35 = 128;
            goto LABEL_67;
          case 7:
            v35 = 8 * (unsigned int)sub_15A9520(v8, 0);
            goto LABEL_67;
          case 0xB:
            v35 = *(_DWORD *)(v21 + 8) >> 8;
            goto LABEL_67;
          case 0xD:
            v35 = 8LL * *(_QWORD *)sub_15A9930(v8, v21);
            goto LABEL_67;
          case 0xE:
            v48 = *(_QWORD *)(v21 + 24);
            v60 = *(_QWORD *)(v21 + 32);
            v41 = (unsigned int)sub_15A9FE0(v8, v48);
            v35 = 8 * v60 * v41 * ((v41 + ((unsigned __int64)(sub_127FA20(v8, v48) + 7) >> 3) - 1) / v41);
            goto LABEL_67;
          case 0xF:
            v35 = 8 * (unsigned int)sub_15A9520(v8, *(_DWORD *)(v21 + 8) >> 8);
LABEL_67:
            *a3 = (v35 * v20 + 7) & 0xFFFFFFFFFFFFFFF8LL;
            *a4 = 0;
            v13 = *(_QWORD *)(a1 - 72);
            if ( !v13 )
              return 0;
            goto LABEL_8;
          case 0x10:
            v42 = *(_QWORD *)(v21 + 32);
            v21 = *(_QWORD *)(v21 + 24);
            v20 *= v42;
            continue;
          default:
LABEL_88:
            BUG();
        }
      }
    }
    return 0;
  }
  if ( !byte_4FAA260 )
    return 0;
  *a2 = 0;
  v10 = *(_QWORD *)a1;
  v11 = 1;
  while ( 2 )
  {
    switch ( *(_BYTE *)(v10 + 8) )
    {
      case 0:
      case 8:
      case 0xA:
      case 0xC:
      case 0x10:
        v27 = *(_QWORD *)(v10 + 32);
        v10 = *(_QWORD *)(v10 + 24);
        v11 *= v27;
        continue;
      case 1:
        v12 = 16;
        break;
      case 2:
        v12 = 32;
        break;
      case 3:
      case 9:
        v12 = 64;
        break;
      case 4:
        v12 = 80;
        break;
      case 5:
      case 6:
        v12 = 128;
        break;
      case 7:
        v12 = 8 * (unsigned int)sub_15A9520(v8, 0);
        break;
      case 0xB:
        v12 = *(_DWORD *)(v10 + 8) >> 8;
        break;
      case 0xD:
        v12 = 8LL * *(_QWORD *)sub_15A9930(v8, v10);
        break;
      case 0xE:
        v22 = *(_QWORD *)(v10 + 32);
        v49 = *(_QWORD *)(v10 + 24);
        v23 = sub_15A9FE0(v8, v49);
        v24 = v49;
        v25 = 1;
        v26 = v23;
        while ( 2 )
        {
          switch ( *(_BYTE *)(v24 + 8) )
          {
            case 0:
            case 8:
            case 0xA:
            case 0xC:
            case 0x10:
              v39 = *(_QWORD *)(v24 + 32);
              v24 = *(_QWORD *)(v24 + 24);
              v25 *= v39;
              continue;
            case 1:
              v32 = 16;
              break;
            case 2:
              v32 = 32;
              break;
            case 3:
            case 9:
              v32 = 64;
              break;
            case 4:
              v32 = 80;
              break;
            case 5:
            case 6:
              v32 = 128;
              break;
            case 7:
              v52 = v25;
              v58 = v26;
              v38 = sub_15A9520(v8, 0);
              v26 = v58;
              v25 = v52;
              v32 = (unsigned int)(8 * v38);
              break;
            case 0xB:
              v32 = *(_DWORD *)(v24 + 8) >> 8;
              break;
            case 0xD:
              v51 = v25;
              v57 = v26;
              v37 = (_QWORD *)sub_15A9930(v8, v24);
              v26 = v57;
              v25 = v51;
              v32 = 8LL * *v37;
              break;
            case 0xE:
              v43 = v25;
              v44 = v26;
              v45 = *(_QWORD *)(v24 + 24);
              v56 = *(_QWORD *)(v24 + 32);
              v50 = (unsigned int)sub_15A9FE0(v8, v45);
              v36 = sub_127FA20(v8, v45);
              v26 = v44;
              v25 = v43;
              v32 = 8 * v56 * v50 * ((v50 + ((unsigned __int64)(v36 + 7) >> 3) - 1) / v50);
              break;
            case 0xF:
              v53 = v25;
              v59 = v26;
              v40 = sub_15A9520(v8, *(_DWORD *)(v24 + 8) >> 8);
              v26 = v59;
              v25 = v53;
              v32 = (unsigned int)(8 * v40);
              break;
          }
          break;
        }
        v12 = 8 * v26 * v22 * ((v26 + ((unsigned __int64)(v32 * v25 + 7) >> 3) - 1) / v26);
        break;
      case 0xF:
        v12 = 8 * (unsigned int)sub_15A9520(v8, *(_DWORD *)(v10 + 8) >> 8);
        break;
    }
    break;
  }
  *a3 = (v11 * v12 + 7) & 0xFFFFFFFFFFFFFFF8LL;
LABEL_7:
  *a4 = 1 << (*(unsigned __int16 *)(a1 + 18) >> 1) >> 1;
  v13 = *(_QWORD *)(a1 - 24);
  if ( !v13 )
    return 0;
LABEL_8:
  v14 = *(_QWORD *)v13;
  if ( *(_BYTE *)(*(_QWORD *)v13 + 8LL) == 16 )
  {
    v14 = **(_QWORD **)(v14 + 16);
    if ( *(_BYTE *)(v14 + 8) == 16 )
      v14 = **(_QWORD **)(v14 + 16);
  }
  if ( *(_DWORD *)(v14 + 8) >> 8 || (unsigned __int8)sub_1649A90(v13) )
    return 0;
  return v13;
}
