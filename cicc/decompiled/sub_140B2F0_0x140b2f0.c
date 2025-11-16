// Function: sub_140B2F0
// Address: 0x140b2f0
//
__int64 __fastcall sub_140B2F0(__int64 a1, __int64 a2, __int64 a3, unsigned __int8 a4)
{
  __int64 v6; // rax
  __int64 v7; // r12
  unsigned __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r14
  __int64 v11; // rsi
  unsigned int v12; // r9d
  __int64 v14; // rax
  __int64 v15; // rsi
  int v16; // edx
  int v17; // eax
  int v18; // eax
  unsigned int v19; // eax
  __int64 v20; // rsi
  __int64 v21; // rdx
  unsigned __int64 v22; // r10
  _QWORD *v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  unsigned __int64 v26; // rax
  _QWORD *v27; // rax
  __int64 v28; // rsi
  int v29; // eax
  __int64 v30; // rax
  __int64 v31; // [rsp-70h] [rbp-70h]
  unsigned __int64 v32; // [rsp-68h] [rbp-68h]
  __int64 v33; // [rsp-68h] [rbp-68h]
  __int64 v34; // [rsp-68h] [rbp-68h]
  unsigned int v35; // [rsp-60h] [rbp-60h]
  unsigned __int64 v36; // [rsp-60h] [rbp-60h]
  unsigned __int64 v37; // [rsp-60h] [rbp-60h]
  __int64 v38; // [rsp-58h] [rbp-58h]
  __int64 v39; // [rsp-58h] [rbp-58h]
  unsigned int v40; // [rsp-58h] [rbp-58h]
  unsigned int v41; // [rsp-58h] [rbp-58h]
  unsigned int v42; // [rsp-50h] [rbp-50h]
  unsigned int v43; // [rsp-50h] [rbp-50h]
  __int64 v44; // [rsp-50h] [rbp-50h]
  unsigned int v45; // [rsp-50h] [rbp-50h]
  __int64 v46; // [rsp-40h] [rbp-40h] BYREF

  if ( !a1 )
    return 0;
  v6 = sub_140B2D0((_QWORD *)a1);
  v7 = v6;
  if ( !v6 )
    return 0;
  v8 = *(unsigned __int8 *)(v6 + 8);
  if ( (unsigned __int8)v8 > 0xFu || (v9 = 35454, !_bittest64(&v9, v8)) )
  {
    if ( (unsigned int)(v8 - 13) > 1 && (_DWORD)v8 != 16 || !(unsigned __int8)sub_16435F0(v7, 0) )
      return 0;
  }
  v10 = 1;
  v11 = v7;
  v12 = sub_15A9FE0(a2, v7);
  while ( 2 )
  {
    switch ( *(_BYTE *)(v11 + 8) )
    {
      case 0:
      case 8:
      case 0xA:
      case 0xC:
      case 0x10:
        v24 = *(_QWORD *)(v11 + 32);
        v11 = *(_QWORD *)(v11 + 24);
        v10 *= v24;
        continue;
      case 1:
        v14 = 16;
        break;
      case 2:
        v14 = 32;
        break;
      case 3:
      case 9:
        v14 = 64;
        break;
      case 4:
        v14 = 80;
        break;
      case 5:
      case 6:
        v14 = 128;
        break;
      case 7:
        v42 = v12;
        v17 = sub_15A9520(a2, 0);
        v12 = v42;
        v14 = (unsigned int)(8 * v17);
        break;
      case 0xB:
        v14 = *(_DWORD *)(v11 + 8) >> 8;
        break;
      case 0xD:
        v45 = v12;
        v23 = (_QWORD *)sub_15A9930(a2, v11);
        v12 = v45;
        v14 = 8LL * *v23;
        break;
      case 0xE:
        v35 = v12;
        v38 = *(_QWORD *)(v11 + 24);
        v44 = *(_QWORD *)(v11 + 32);
        v19 = sub_15A9FE0(a2, v38);
        v12 = v35;
        v20 = v38;
        v21 = 1;
        v22 = v19;
        while ( 2 )
        {
          switch ( *(_BYTE *)(v20 + 8) )
          {
            case 0:
            case 8:
            case 0xA:
            case 0xC:
            case 0x10:
              v30 = *(_QWORD *)(v20 + 32);
              v20 = *(_QWORD *)(v20 + 24);
              v21 *= v30;
              continue;
            case 1:
              v25 = 16;
              goto LABEL_30;
            case 2:
              v25 = 32;
              goto LABEL_30;
            case 3:
            case 9:
              v25 = 64;
              goto LABEL_30;
            case 4:
              v25 = 80;
              goto LABEL_30;
            case 5:
            case 6:
              v25 = 128;
              goto LABEL_30;
            case 7:
              v34 = v21;
              v28 = 0;
              v37 = v22;
              v41 = v12;
              goto LABEL_37;
            case 0xB:
              JUMPOUT(0x140B5DC);
            case 0xD:
              v33 = v21;
              v36 = v22;
              v40 = v12;
              v27 = (_QWORD *)sub_15A9930(a2, v20);
              v12 = v40;
              v22 = v36;
              v21 = v33;
              v25 = 8LL * *v27;
              goto LABEL_30;
            case 0xE:
              v31 = v21;
              v32 = v22;
              v39 = *(_QWORD *)(v20 + 32);
              v26 = sub_12BE0A0(a2, *(_QWORD *)(v20 + 24));
              v12 = v35;
              v22 = v32;
              v21 = v31;
              v25 = 8 * v39 * v26;
              goto LABEL_30;
            case 0xF:
              v34 = v21;
              v37 = v22;
              v41 = v12;
              v28 = *(_DWORD *)(v20 + 8) >> 8;
LABEL_37:
              v29 = sub_15A9520(a2, v28);
              v12 = v41;
              v22 = v37;
              v21 = v34;
              v25 = (unsigned int)(8 * v29);
LABEL_30:
              v14 = 8 * v44 * v22 * ((v22 + ((unsigned __int64)(v25 * v21 + 7) >> 3) - 1) / v22);
              break;
          }
          break;
        }
        break;
      case 0xF:
        v43 = v12;
        v18 = sub_15A9520(a2, *(_DWORD *)(v11 + 8) >> 8);
        v12 = v43;
        v14 = (unsigned int)(8 * v18);
        break;
    }
    break;
  }
  v15 = *(_BYTE *)(v7 + 8) == 13
      ? *(unsigned int *)sub_15A9930(a2, v7)
      : (unsigned int)((v12 + ((unsigned __int64)(v10 * v14 + 7) >> 3) - 1) / v12) * v12;
  v16 = *(_DWORD *)(a1 + 20);
  v46 = 0;
  if ( (unsigned __int8)sub_14AAC00(*(_QWORD *)(a1 - 24LL * (v16 & 0xFFFFFFF)), v15, &v46, a4, 0) )
    return v46;
  else
    return 0;
}
