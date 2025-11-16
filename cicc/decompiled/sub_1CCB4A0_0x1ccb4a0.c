// Function: sub_1CCB4A0
// Address: 0x1ccb4a0
//
__int64 __fastcall sub_1CCB4A0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r12
  __int64 *v4; // rbx
  __int64 v5; // r14
  _QWORD *v6; // rax
  __int64 v7; // r13
  __int64 v8; // r13
  unsigned int v9; // eax
  __int64 v10; // rsi
  __int64 v11; // rcx
  unsigned __int64 v12; // r9
  __int64 v14; // rax
  unsigned int v15; // esi
  __int64 *v16; // rdx
  unsigned __int64 v17; // rax
  __int64 v18; // r9
  _QWORD *v19; // rax
  unsigned int v20; // eax
  __int64 v21; // rsi
  __int64 v22; // r10
  unsigned __int64 v23; // r11
  unsigned int v24; // esi
  int v25; // eax
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rax
  _QWORD *v29; // rax
  int v30; // eax
  __int64 v31; // rax
  __int64 v32; // [rsp+8h] [rbp-78h]
  __int64 v33; // [rsp+10h] [rbp-70h]
  unsigned __int64 v34; // [rsp+18h] [rbp-68h]
  __int64 v35; // [rsp+20h] [rbp-60h]
  __int64 v36; // [rsp+20h] [rbp-60h]
  __int64 v37; // [rsp+20h] [rbp-60h]
  __int64 v38; // [rsp+28h] [rbp-58h]
  __int64 v39; // [rsp+28h] [rbp-58h]
  __int64 v40; // [rsp+28h] [rbp-58h]
  __int64 v41; // [rsp+28h] [rbp-58h]
  __int64 v42; // [rsp+30h] [rbp-50h]
  unsigned __int64 v43; // [rsp+30h] [rbp-50h]
  unsigned __int64 v44; // [rsp+30h] [rbp-50h]
  unsigned __int64 v45; // [rsp+30h] [rbp-50h]
  unsigned __int64 v46; // [rsp+38h] [rbp-48h]
  __int64 v47; // [rsp+38h] [rbp-48h]
  unsigned __int64 v48; // [rsp+38h] [rbp-48h]
  __int64 v49; // [rsp+38h] [rbp-48h]
  __int64 v50; // [rsp+38h] [rbp-48h]
  __int64 v51; // [rsp+38h] [rbp-48h]
  int v52; // [rsp+40h] [rbp-40h]
  __int64 v53; // [rsp+40h] [rbp-40h]
  __int64 v54; // [rsp+40h] [rbp-40h]
  __int64 v55; // [rsp+40h] [rbp-40h]

  v2 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
  v3 = **(_QWORD **)(a1 - 24 * v2);
  v4 = (__int64 *)(a1 + 24 * (1 - v2));
  if ( (__int64 *)a1 != v4 )
  {
    v5 = 0;
    while ( 1 )
    {
      v8 = *v4;
      if ( *(_BYTE *)(v3 + 8) == 13 )
      {
        v6 = *(_QWORD **)(v8 + 24);
        if ( *(_DWORD *)(v8 + 32) > 0x40u )
          v6 = (_QWORD *)*v6;
        v7 = (unsigned int)v6;
        if ( (_DWORD)v6 )
          v5 += *(_QWORD *)(sub_15A9930(a2, v3) + 8LL * (unsigned int)v6 + 16);
        v3 = *(_QWORD *)(*(_QWORD *)(v3 + 16) + 8 * v7);
        goto LABEL_8;
      }
      v3 = *(_QWORD *)(v3 + 24);
      if ( *(_BYTE *)(v8 + 16) == 13 )
      {
        if ( *(_DWORD *)(v8 + 32) <= 0x40u )
        {
          if ( *(_QWORD *)(v8 + 24) )
          {
LABEL_13:
            v9 = sub_15A9FE0(a2, v3);
            v10 = v3;
            v11 = 1;
            v12 = v9;
            while ( 2 )
            {
              switch ( *(_BYTE *)(v10 + 8) )
              {
                case 0:
                case 8:
                case 0xA:
                case 0xC:
                case 0x10:
                  v26 = *(_QWORD *)(v10 + 32);
                  v10 = *(_QWORD *)(v10 + 24);
                  v11 *= v26;
                  continue;
                case 1:
                  v14 = 16;
                  goto LABEL_20;
                case 2:
                  v14 = 32;
                  goto LABEL_20;
                case 3:
                case 9:
                  v14 = 64;
                  goto LABEL_20;
                case 4:
                  v14 = 80;
                  goto LABEL_20;
                case 5:
                case 6:
                  v14 = 128;
                  goto LABEL_20;
                case 7:
                  v48 = v12;
                  v24 = 0;
                  v55 = v11;
                  goto LABEL_28;
                case 0xB:
                  v14 = *(_DWORD *)(v10 + 8) >> 8;
                  goto LABEL_20;
                case 0xD:
                  v46 = v12;
                  v53 = v11;
                  v19 = (_QWORD *)sub_15A9930(a2, v10);
                  v11 = v53;
                  v12 = v46;
                  v14 = 8LL * *v19;
                  goto LABEL_20;
                case 0xE:
                  v38 = v12;
                  v42 = v11;
                  v54 = *(_QWORD *)(v10 + 32);
                  v47 = *(_QWORD *)(v10 + 24);
                  v20 = sub_15A9FE0(a2, v47);
                  v12 = v38;
                  v21 = v47;
                  v22 = 1;
                  v11 = v42;
                  v23 = v20;
                  while ( 2 )
                  {
                    switch ( *(_BYTE *)(v21 + 8) )
                    {
                      case 0:
                      case 8:
                      case 0xA:
                      case 0xC:
                      case 0x10:
                        v31 = *(_QWORD *)(v21 + 32);
                        v21 = *(_QWORD *)(v21 + 24);
                        v22 *= v31;
                        continue;
                      case 1:
                        v27 = 16;
                        goto LABEL_37;
                      case 2:
                        v27 = 32;
                        goto LABEL_37;
                      case 3:
                      case 9:
                        v27 = 64;
                        goto LABEL_37;
                      case 4:
                        v27 = 80;
                        goto LABEL_37;
                      case 5:
                      case 6:
                        v27 = 128;
                        goto LABEL_37;
                      case 7:
                        JUMPOUT(0x1CCB80B);
                      case 0xB:
                        v27 = *(_DWORD *)(v21 + 8) >> 8;
                        goto LABEL_37;
                      case 0xD:
                        v36 = v38;
                        v40 = v42;
                        v44 = v23;
                        v50 = v22;
                        v29 = (_QWORD *)sub_15A9930(a2, v21);
                        v22 = v50;
                        v23 = v44;
                        v11 = v40;
                        v12 = v36;
                        v27 = 8LL * *v29;
                        goto LABEL_37;
                      case 0xE:
                        v32 = v38;
                        v33 = v42;
                        v34 = v23;
                        v35 = v22;
                        v39 = *(_QWORD *)(v21 + 24);
                        v49 = *(_QWORD *)(v21 + 32);
                        v43 = (unsigned int)sub_15A9FE0(a2, v39);
                        v28 = sub_127FA20(a2, v39);
                        v22 = v35;
                        v23 = v34;
                        v11 = v33;
                        v12 = v32;
                        v27 = 8 * v43 * v49 * ((v43 + ((unsigned __int64)(v28 + 7) >> 3) - 1) / v43);
                        goto LABEL_37;
                      case 0xF:
                        v37 = v38;
                        v41 = v42;
                        v45 = v23;
                        v51 = v22;
                        v30 = sub_15A9520(a2, *(_DWORD *)(v21 + 8) >> 8);
                        v22 = v51;
                        v23 = v45;
                        v11 = v41;
                        v12 = v37;
                        v27 = (unsigned int)(8 * v30);
LABEL_37:
                        v14 = 8 * v54 * v23 * ((v23 + ((unsigned __int64)(v27 * v22 + 7) >> 3) - 1) / v23);
                        break;
                    }
                    goto LABEL_20;
                  }
                case 0xF:
                  v48 = v12;
                  v55 = v11;
                  v24 = *(_DWORD *)(v10 + 8) >> 8;
LABEL_28:
                  v25 = sub_15A9520(a2, v24);
                  v11 = v55;
                  v12 = v48;
                  v14 = (unsigned int)(8 * v25);
LABEL_20:
                  v15 = *(_DWORD *)(v8 + 32);
                  v16 = *(__int64 **)(v8 + 24);
                  v17 = v12 * ((v12 + ((unsigned __int64)(v14 * v11 + 7) >> 3) - 1) / v12);
                  if ( v15 > 0x40 )
                    v18 = *v16;
                  else
                    v18 = (__int64)((_QWORD)v16 << (64 - (unsigned __int8)v15)) >> (64 - (unsigned __int8)v15);
                  v5 += v18 * v17;
                  break;
              }
              break;
            }
          }
        }
        else
        {
          v52 = *(_DWORD *)(v8 + 32);
          if ( v52 != (unsigned int)sub_16A57B0(v8 + 24) )
            goto LABEL_13;
        }
      }
LABEL_8:
      v4 += 3;
      if ( (__int64 *)a1 == v4 )
        return v5;
    }
  }
  return 0;
}
