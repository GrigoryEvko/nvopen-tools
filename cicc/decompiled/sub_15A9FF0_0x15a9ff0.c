// Function: sub_15A9FF0
// Address: 0x15a9ff0
//
__int64 __fastcall sub_15A9FF0(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4)
{
  __int64 v4; // rsi
  __int64 v5; // r12
  __int64 v6; // r15
  _QWORD *v7; // r14
  __int64 v8; // r8
  unsigned __int64 v9; // r12
  __int64 v10; // r9
  unsigned int v11; // eax
  __int64 v12; // rbx
  int v13; // r8d
  char v14; // al
  __int64 v16; // rbx
  __int64 v17; // rsi
  unsigned int v18; // eax
  __int64 v19; // rcx
  unsigned __int64 v20; // r11
  __int64 v21; // rax
  __int64 v22; // rax
  _QWORD *v23; // rax
  unsigned int v24; // eax
  __int64 v25; // rsi
  unsigned __int64 v26; // r10
  unsigned int v27; // esi
  int v28; // eax
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rax
  char v32; // [rsp+20h] [rbp-70h]
  unsigned __int64 v33; // [rsp+28h] [rbp-68h]
  __int64 v34; // [rsp+30h] [rbp-60h]
  char v35; // [rsp+38h] [rbp-58h]
  __int64 v36; // [rsp+38h] [rbp-58h]
  char v37; // [rsp+38h] [rbp-58h]
  __int64 v38; // [rsp+40h] [rbp-50h]
  unsigned __int64 v39; // [rsp+40h] [rbp-50h]
  __int64 v40; // [rsp+40h] [rbp-50h]
  unsigned __int64 v41; // [rsp+40h] [rbp-50h]
  char v42; // [rsp+48h] [rbp-48h]
  __int64 v43; // [rsp+48h] [rbp-48h]
  __int64 v44; // [rsp+48h] [rbp-48h]
  __int64 v45; // [rsp+48h] [rbp-48h]
  char v46; // [rsp+50h] [rbp-40h]
  __int64 v47; // [rsp+50h] [rbp-40h]
  __int64 v48; // [rsp+50h] [rbp-40h]
  __int64 v49; // [rsp+50h] [rbp-40h]
  _QWORD *v50; // [rsp+58h] [rbp-38h]

  v4 = a2 | 4;
  v50 = &a3[a4];
  if ( v50 != a3 )
  {
    v5 = v4;
    v6 = 0;
    v7 = a3;
    while ( 1 )
    {
      v8 = v5;
      v9 = v5 & 0xFFFFFFFFFFFFFFF8LL;
      v10 = v9;
      v11 = *(_DWORD *)(*v7 + 32LL);
      v12 = *(_QWORD *)(*v7 + 24LL);
      v13 = (v8 >> 2) & 1;
      if ( v13 || !v9 )
        break;
      if ( v11 > 0x40 )
        v12 = *(_QWORD *)v12;
      v6 += *(_QWORD *)(sub_15A9930(a1, v9) + 8LL * (unsigned int)v12 + 16);
LABEL_9:
      v10 = sub_1643D30(v9, *v7);
LABEL_10:
      v14 = *(_BYTE *)(v10 + 8);
      if ( ((v14 - 14) & 0xFD) != 0 )
      {
        v5 = 0;
        if ( v14 == 13 )
          v5 = v10;
        if ( v50 == ++v7 )
          return v6;
      }
      else
      {
        ++v7;
        v5 = *(_QWORD *)(v10 + 24) | 4LL;
        if ( v50 == v7 )
          return v6;
      }
    }
    if ( v11 > 0x40 )
    {
      v16 = *(_QWORD *)v12;
      if ( !v16 )
        goto LABEL_17;
    }
    else
    {
      v16 = v12 << (64 - (unsigned __int8)v11) >> (64 - (unsigned __int8)v11);
      if ( !v16 )
        goto LABEL_17;
    }
    if ( !(_BYTE)v13 || (v17 = v9) == 0 )
    {
      v46 = v13;
      v21 = sub_1643D30(v9, *v7);
      v10 = v9;
      LOBYTE(v13) = v46;
      v17 = v21;
    }
    v38 = v10;
    v42 = v13;
    v18 = sub_15A9FE0(a1, v17);
    LOBYTE(v13) = v42;
    v19 = 1;
    v10 = v38;
    v20 = v18;
    while ( 2 )
    {
      switch ( *(_BYTE *)(v17 + 8) )
      {
        case 0:
        case 8:
        case 0xA:
        case 0xC:
        case 0x10:
          v29 = *(_QWORD *)(v17 + 32);
          v17 = *(_QWORD *)(v17 + 24);
          v19 *= v29;
          continue;
        case 1:
          v22 = 16;
          goto LABEL_27;
        case 2:
          v22 = 32;
          goto LABEL_27;
        case 3:
        case 9:
          v22 = 64;
          goto LABEL_27;
        case 4:
          v22 = 80;
          goto LABEL_27;
        case 5:
        case 6:
          v22 = 128;
          goto LABEL_27;
        case 7:
          v37 = v42;
          v27 = 0;
          v41 = v20;
          v45 = v19;
          v49 = v10;
          goto LABEL_33;
        case 0xB:
          v22 = *(_DWORD *)(v17 + 8) >> 8;
          goto LABEL_27;
        case 0xD:
          v35 = v42;
          v39 = v20;
          v43 = v19;
          v47 = v10;
          v23 = (_QWORD *)sub_15A9930(a1, v17);
          v10 = v47;
          v19 = v43;
          v20 = v39;
          LOBYTE(v13) = v35;
          v22 = 8LL * *v23;
          goto LABEL_27;
        case 0xE:
          v32 = v42;
          v33 = v20;
          v34 = v19;
          v36 = v38;
          v40 = *(_QWORD *)(v17 + 24);
          v44 = *(_QWORD *)(v17 + 32);
          v24 = sub_15A9FE0(a1, v40);
          LOBYTE(v13) = v32;
          v20 = v33;
          v48 = 1;
          v25 = v40;
          v19 = v34;
          v26 = v24;
          v10 = v36;
          while ( 2 )
          {
            switch ( *(_BYTE *)(v25 + 8) )
            {
              case 0:
                v31 = v48 * *(_QWORD *)(v25 + 32);
                v25 = *(_QWORD *)(v25 + 24);
                v48 = v31;
                continue;
              case 1:
                JUMPOUT(0x15AA33F);
              case 2:
                v30 = 32;
                goto LABEL_42;
              case 3:
                v30 = 64;
                goto LABEL_42;
              case 4:
                v30 = 80;
                goto LABEL_42;
              case 5:
              case 6:
                v30 = 128;
LABEL_42:
                v22 = 8 * v26 * v44 * ((v26 + ((unsigned __int64)(v48 * v30 + 7) >> 3) - 1) / v26);
                break;
            }
            goto LABEL_27;
          }
        case 0xF:
          v37 = v42;
          v41 = v20;
          v45 = v19;
          v27 = *(_DWORD *)(v17 + 8) >> 8;
          v49 = v10;
LABEL_33:
          v28 = sub_15A9520(a1, v27);
          v10 = v49;
          v19 = v45;
          v20 = v41;
          LOBYTE(v13) = v37;
          v22 = (unsigned int)(8 * v28);
LABEL_27:
          v6 += v20 * v16 * ((v20 + ((unsigned __int64)(v22 * v19 + 7) >> 3) - 1) / v20);
          break;
      }
      break;
    }
LABEL_17:
    if ( (_BYTE)v13 && v9 )
      goto LABEL_10;
    goto LABEL_9;
  }
  return 0;
}
