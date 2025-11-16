// Function: sub_1AEBB60
// Address: 0x1aebb60
//
__int64 __fastcall sub_1AEBB60(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v6; // r13
  __int64 v7; // rbx
  __int64 v10; // rax
  __int64 v11; // rcx
  __int64 v12; // rax
  char v13; // dl
  __int64 v14; // rsi
  void (__fastcall *v15)(__int64 *, __int64, __int64); // r8
  char *v16; // r9
  __int64 v17; // r11
  __int64 v18; // r9
  char v19; // r11
  _DWORD *v20; // rdi
  _DWORD *v21; // rsi
  char v22; // r8
  _DWORD *v23; // rdi
  _DWORD *v24; // rsi
  unsigned int v25; // r13d
  unsigned int v26; // esi
  int v27; // eax
  __int64 v28; // rax
  __int64 v29; // rax
  _QWORD *v30; // rax
  unsigned int v31; // esi
  int v32; // eax
  __int64 v33; // rax
  __int64 v34; // rax
  _QWORD *v35; // rax
  __int64 v36; // [rsp+0h] [rbp-90h]
  unsigned __int64 v37; // [rsp+8h] [rbp-88h]
  __int64 v38; // [rsp+10h] [rbp-80h]
  __int64 v39; // [rsp+10h] [rbp-80h]
  __int64 v40; // [rsp+18h] [rbp-78h]
  __int64 v41; // [rsp+18h] [rbp-78h]
  __int64 v42; // [rsp+18h] [rbp-78h]
  __int64 v43; // [rsp+18h] [rbp-78h]
  __int64 v44; // [rsp+18h] [rbp-78h]
  __int64 v45; // [rsp+20h] [rbp-70h]
  unsigned __int64 v46; // [rsp+20h] [rbp-70h]
  __int64 v47; // [rsp+28h] [rbp-68h]
  __int64 v48; // [rsp+30h] [rbp-60h]
  __int64 v49; // [rsp+30h] [rbp-60h]
  __int64 v50; // [rsp+30h] [rbp-60h]
  __int64 v51; // [rsp+30h] [rbp-60h]
  char v53; // [rsp+47h] [rbp-49h] BYREF
  unsigned __int64 v54; // [rsp+48h] [rbp-48h] BYREF
  _QWORD v55[8]; // [rsp+50h] [rbp-40h] BYREF

  if ( (*(_BYTE *)(a1 + 23) & 0x10) == 0 )
    return 0;
  v6 = *(_QWORD *)a1;
  v7 = *a2;
  v10 = sub_15F2050(a1);
  v11 = sub_1632FA0(v10);
  if ( v7 == v6 )
  {
LABEL_8:
    v15 = (void (__fastcall *)(__int64 *, __int64, __int64))sub_1AE75D0;
    v16 = &v53;
  }
  else
  {
    v12 = *(unsigned __int8 *)(v6 + 8);
    if ( (*(_BYTE *)(v6 + 8) & 0xFB) != 0xB )
      return 0;
    v13 = *(_BYTE *)(v7 + 8);
    if ( (v13 & 0xFB) != 0xB )
      return 0;
    v47 = 1;
    v14 = v6;
    while ( 2 )
    {
      switch ( v12 )
      {
        case 0LL:
        case 8LL:
        case 10LL:
        case 12LL:
        case 16LL:
          v28 = v47 * *(_QWORD *)(v14 + 32);
          v14 = *(_QWORD *)(v14 + 24);
          v47 = v28;
          v12 = *(unsigned __int8 *)(v14 + 8);
          continue;
        case 1LL:
          v45 = 16;
          goto LABEL_12;
        case 2LL:
          v45 = 32;
          goto LABEL_12;
        case 3LL:
        case 9LL:
          v45 = 64;
          goto LABEL_12;
        case 4LL:
          v45 = 80;
          goto LABEL_12;
        case 5LL:
        case 6LL:
          v45 = 128;
          goto LABEL_12;
        case 7LL:
          v26 = 0;
          goto LABEL_30;
        case 11LL:
          v45 = *(_DWORD *)(v14 + 8) >> 8;
          goto LABEL_12;
        case 13LL:
          v51 = v11;
          v30 = (_QWORD *)sub_15A9930(v11, v14);
          v13 = *(_BYTE *)(v7 + 8);
          v11 = v51;
          v45 = 8LL * *v30;
          goto LABEL_12;
        case 14LL:
          v41 = v11;
          v38 = *(_QWORD *)(v14 + 24);
          v50 = *(_QWORD *)(v14 + 32);
          v46 = (unsigned int)sub_15A9FE0(v11, v38);
          v29 = sub_127FA20(v41, v38);
          v11 = v41;
          v13 = *(_BYTE *)(v7 + 8);
          v45 = 8 * v50 * v46 * ((v46 + ((unsigned __int64)(v29 + 7) >> 3) - 1) / v46);
          goto LABEL_12;
        case 15LL:
          v26 = *(_DWORD *)(v14 + 8) >> 8;
LABEL_30:
          v49 = v11;
          v27 = sub_15A9520(v11, v26);
          v13 = *(_BYTE *)(v7 + 8);
          v11 = v49;
          v45 = (unsigned int)(8 * v27);
LABEL_12:
          v48 = 1;
          v17 = v7;
          while ( 2 )
          {
            switch ( v13 )
            {
              case 1:
                v18 = 16;
                goto LABEL_16;
              case 2:
                v18 = 32;
                goto LABEL_16;
              case 3:
              case 9:
                v18 = 64;
                goto LABEL_16;
              case 4:
                v18 = 80;
                goto LABEL_16;
              case 5:
              case 6:
                v18 = 128;
                goto LABEL_16;
              case 7:
                v31 = 0;
                goto LABEL_41;
              case 11:
                v18 = *(_DWORD *)(v17 + 8) >> 8;
                goto LABEL_16;
              case 13:
                v44 = v11;
                v35 = (_QWORD *)sub_15A9930(v11, v17);
                v11 = v44;
                v18 = 8LL * *v35;
                goto LABEL_16;
              case 14:
                v39 = v11;
                v36 = *(_QWORD *)(v17 + 24);
                v43 = *(_QWORD *)(v17 + 32);
                v37 = (unsigned int)sub_15A9FE0(v11, v36);
                v34 = sub_127FA20(v39, v36);
                v11 = v39;
                v18 = 8 * v43 * v37 * ((v37 + ((unsigned __int64)(v34 + 7) >> 3) - 1) / v37);
                goto LABEL_16;
              case 15:
                v31 = *(_DWORD *)(v17 + 8) >> 8;
LABEL_41:
                v42 = v11;
                v32 = sub_15A9520(v11, v31);
                v11 = v42;
                v18 = (unsigned int)(8 * v32);
LABEL_16:
                v19 = *(_BYTE *)(v6 + 8);
                if ( v19 != 15 )
                  goto LABEL_19;
                v20 = *(_DWORD **)(v11 + 408);
                v40 = v11;
                v21 = &v20[*(unsigned int *)(v11 + 416)];
                LODWORD(v55[0]) = *(_DWORD *)(v6 + 8) >> 8;
                if ( v21 != sub_1AE7760(v20, (__int64)v21, (int *)v55) )
                  return 0;
                v11 = v40;
LABEL_19:
                v22 = *(_BYTE *)(v7 + 8);
                if ( v22 == 15 )
                {
                  v23 = *(_DWORD **)(v11 + 408);
                  v24 = &v23[*(unsigned int *)(v11 + 416)];
                  LODWORD(v55[0]) = *(_DWORD *)(v7 + 8) >> 8;
                  if ( v24 != sub_1AE7760(v23, (__int64)v24, (int *)v55) )
                    return 0;
                }
                if ( v18 * v48 == v45 * v47 )
                  goto LABEL_8;
                if ( v19 != 11 || v22 != 11 )
                  return 0;
                v25 = sub_1643030(v6);
                v54 = (unsigned int)sub_1643030(v7);
                if ( v54 > v25 )
                  goto LABEL_8;
                v15 = (void (__fastcall *)(__int64 *, __int64, __int64))sub_1AE7CC0;
                v55[0] = &v53;
                v16 = (char *)v55;
                v55[1] = &v54;
                break;
              case 16:
                v33 = v48 * *(_QWORD *)(v17 + 32);
                v17 = *(_QWORD *)(v17 + 24);
                v48 = v33;
                v13 = *(_BYTE *)(v17 + 8);
                continue;
              default:
                BUG();
            }
            return result;
          }
      }
    }
  }
  return sub_1AEB5C0(a1, (__int64)a2, a3, a4, v15, (__int64)v16);
}
