// Function: sub_18B8B00
// Address: 0x18b8b00
//
__int64 __fastcall sub_18B8B00(__int64 *a1, __int64 a2, unsigned __int64 a3)
{
  __int64 v3; // r14
  __int64 v6; // rdi
  char v7; // al
  unsigned __int64 *v8; // rax
  unsigned __int64 *v9; // r15
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 result; // rax
  unsigned int v13; // eax
  __int64 v14; // rsi
  __int64 v15; // r9
  unsigned __int64 v16; // r15
  __int64 v17; // rax
  __int64 v18; // rsi
  unsigned __int64 v19; // r8
  unsigned __int64 v20; // rax
  __int64 v21; // rcx
  int v22; // eax
  int v23; // eax
  _QWORD *v24; // rax
  unsigned int v25; // eax
  __int64 v26; // rsi
  __int64 v27; // rdx
  unsigned __int64 v28; // r10
  __int64 v29; // rax
  __int64 v30; // rax
  unsigned int v31; // esi
  int v32; // eax
  __int64 v33; // rax
  unsigned __int64 v34; // rax
  _QWORD *v35; // rax
  __int64 v36; // [rsp+8h] [rbp-58h]
  __int64 v37; // [rsp+10h] [rbp-50h]
  __int64 v38; // [rsp+10h] [rbp-50h]
  unsigned __int64 v39; // [rsp+10h] [rbp-50h]
  __int64 v40; // [rsp+10h] [rbp-50h]
  __int64 v41; // [rsp+18h] [rbp-48h]
  unsigned __int64 v42; // [rsp+18h] [rbp-48h]
  __int64 v43; // [rsp+18h] [rbp-48h]
  unsigned __int64 v44; // [rsp+18h] [rbp-48h]
  __int64 v45; // [rsp+20h] [rbp-40h]
  __int64 v46; // [rsp+20h] [rbp-40h]
  __int64 v47; // [rsp+20h] [rbp-40h]
  __int64 v48; // [rsp+20h] [rbp-40h]
  __int64 v49; // [rsp+28h] [rbp-38h]
  __int64 v50; // [rsp+28h] [rbp-38h]
  __int64 v51; // [rsp+28h] [rbp-38h]
  __int64 v52; // [rsp+28h] [rbp-38h]

  v3 = a2;
  while ( *(_BYTE *)(*(_QWORD *)v3 + 8LL) != 15 )
  {
    v6 = sub_1632FA0(*a1);
    v7 = *(_BYTE *)(v3 + 16);
    if ( v7 == 7 )
    {
      v8 = (unsigned __int64 *)sub_15A9930(v6, *(_QWORD *)v3);
      v9 = v8;
      if ( *v8 <= a3 )
        return 0;
      v10 = (unsigned int)sub_15A8020((__int64)v8, a3);
      a3 -= v9[v10 + 2];
      if ( (*(_BYTE *)(v3 + 23) & 0x40) != 0 )
        v11 = *(_QWORD *)(v3 - 8);
      else
        v11 = v3 - 24LL * (*(_DWORD *)(v3 + 20) & 0xFFFFFFF);
      v3 = *(_QWORD *)(v11 + 24 * v10);
    }
    else
    {
      if ( v7 != 6 )
        return 0;
      v45 = *(_QWORD *)(*(_QWORD *)v3 + 24LL);
      v13 = sub_15A9FE0(v6, v45);
      v14 = v45;
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
            v29 = *(_QWORD *)(v14 + 32);
            v14 = *(_QWORD *)(v14 + 24);
            v15 *= v29;
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
            v49 = v15;
            v22 = sub_15A9520(v6, 0);
            v15 = v49;
            v17 = (unsigned int)(8 * v22);
            break;
          case 0xB:
            v17 = *(_DWORD *)(v14 + 8) >> 8;
            break;
          case 0xD:
            v51 = v15;
            v24 = (_QWORD *)sub_15A9930(v6, v14);
            v15 = v51;
            v17 = 8LL * *v24;
            break;
          case 0xE:
            v37 = v15;
            v41 = *(_QWORD *)(v14 + 24);
            v52 = *(_QWORD *)(v14 + 32);
            v25 = sub_15A9FE0(v6, v41);
            v15 = v37;
            v26 = v41;
            v27 = 1;
            v28 = v25;
            while ( 2 )
            {
              switch ( *(_BYTE *)(v26 + 8) )
              {
                case 0:
                case 8:
                case 0xA:
                case 0xC:
                case 0x10:
                  v33 = *(_QWORD *)(v26 + 32);
                  v26 = *(_QWORD *)(v26 + 24);
                  v27 *= v33;
                  continue;
                case 1:
                  v30 = 16;
                  goto LABEL_33;
                case 2:
                  v30 = 32;
                  goto LABEL_33;
                case 3:
                case 9:
                  v30 = 64;
                  goto LABEL_33;
                case 4:
                  v30 = 80;
                  goto LABEL_33;
                case 5:
                case 6:
                  v30 = 128;
                  goto LABEL_33;
                case 7:
                  v38 = v27;
                  v31 = 0;
                  v42 = v28;
                  v46 = v15;
                  goto LABEL_39;
                case 0xB:
                  v30 = *(_DWORD *)(v26 + 8) >> 8;
                  goto LABEL_33;
                case 0xD:
                  v40 = v27;
                  v44 = v28;
                  v48 = v15;
                  v35 = (_QWORD *)sub_15A9930(v6, v26);
                  v15 = v48;
                  v28 = v44;
                  v27 = v40;
                  v30 = 8LL * *v35;
                  goto LABEL_33;
                case 0xE:
                  v36 = v27;
                  v39 = v28;
                  v43 = v15;
                  v47 = *(_QWORD *)(v26 + 32);
                  v34 = sub_12BE0A0(v6, *(_QWORD *)(v26 + 24));
                  v15 = v43;
                  v28 = v39;
                  v27 = v36;
                  v30 = 8 * v47 * v34;
                  goto LABEL_33;
                case 0xF:
                  v38 = v27;
                  v42 = v28;
                  v46 = v15;
                  v31 = *(_DWORD *)(v26 + 8) >> 8;
LABEL_39:
                  v32 = sub_15A9520(v6, v31);
                  v15 = v46;
                  v28 = v42;
                  v27 = v38;
                  v30 = (unsigned int)(8 * v32);
LABEL_33:
                  v17 = 8 * v52 * v28 * ((v28 + ((unsigned __int64)(v30 * v27 + 7) >> 3) - 1) / v28);
                  break;
              }
              break;
            }
            break;
          case 0xF:
            v50 = v15;
            v23 = sub_15A9520(v6, *(_DWORD *)(v14 + 8) >> 8);
            v15 = v50;
            v17 = (unsigned int)(8 * v23);
            break;
        }
        break;
      }
      v18 = *(_DWORD *)(v3 + 20) & 0xFFFFFFF;
      v19 = v16 * ((v16 + ((unsigned __int64)(v15 * v17 + 7) >> 3) - 1) / v16);
      v20 = a3 / v19;
      if ( (unsigned int)v18 <= (unsigned int)(a3 / v19) )
        return 0;
      a3 %= v19;
      if ( (*(_BYTE *)(v3 + 23) & 0x40) != 0 )
        v21 = *(_QWORD *)(v3 - 8);
      else
        v21 = v3 - 24 * v18;
      v3 = *(_QWORD *)(v21 + 24LL * (unsigned int)v20);
    }
  }
  result = v3;
  if ( a3 )
    return 0;
  return result;
}
