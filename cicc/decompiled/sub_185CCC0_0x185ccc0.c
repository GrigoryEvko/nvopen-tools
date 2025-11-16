// Function: sub_185CCC0
// Address: 0x185ccc0
//
char __fastcall sub_185CCC0(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // r15
  __int64 v4; // r12
  char result; // al
  __int64 v6; // rdi
  __int64 v7; // r13
  __int64 v8; // rsi
  __int64 v9; // rax
  __int64 v10; // rbx
  unsigned __int64 v11; // r13
  __int64 v12; // rax
  _QWORD *v13; // rax
  __int64 v14; // r14
  unsigned int v15; // eax
  __int64 v16; // rsi
  __int64 v17; // rdi
  __int64 v18; // rcx
  int v19; // eax
  int v20; // eax
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // r14
  __int64 v24; // rsi
  unsigned __int64 v25; // r12
  __int64 v26; // rax
  __int64 v27; // rax
  int v28; // eax
  __int64 v29; // rax
  _QWORD *v30; // rax
  int v31; // eax
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // [rsp+8h] [rbp-58h]
  __int64 v35; // [rsp+10h] [rbp-50h]
  __int64 v36; // [rsp+10h] [rbp-50h]
  __int64 v37; // [rsp+18h] [rbp-48h]
  __int64 v38; // [rsp+20h] [rbp-40h]
  unsigned __int64 v39; // [rsp+20h] [rbp-40h]
  unsigned __int64 v40; // [rsp+20h] [rbp-40h]
  __int64 v41; // [rsp+28h] [rbp-38h]
  __int64 v42; // [rsp+28h] [rbp-38h]
  __int64 v43; // [rsp+28h] [rbp-38h]
  __int64 v44; // [rsp+28h] [rbp-38h]
  __int64 v45; // [rsp+28h] [rbp-38h]
  __int64 v46; // [rsp+28h] [rbp-38h]

  v4 = **(_QWORD **)(a2 - 48);
  result = sub_15CCEE0(*(_QWORD *)a1, a2, **(_QWORD **)(a1 + 8));
  if ( result )
  {
    v6 = *(_QWORD *)(a1 + 16);
    v7 = 1;
    v8 = **(_QWORD **)(a1 + 24);
    while ( 2 )
    {
      switch ( *(_BYTE *)(v8 + 8) )
      {
        case 0:
        case 8:
        case 0xA:
        case 0xC:
        case 0x10:
          v21 = *(_QWORD *)(v8 + 32);
          v8 = *(_QWORD *)(v8 + 24);
          v7 *= v21;
          continue;
        case 1:
          v9 = 16;
          goto LABEL_6;
        case 2:
          v9 = 32;
          goto LABEL_6;
        case 3:
        case 9:
          v9 = 64;
          goto LABEL_6;
        case 4:
          v9 = 80;
          goto LABEL_6;
        case 5:
        case 6:
          v9 = 128;
          goto LABEL_6;
        case 7:
          v20 = sub_15A9520(v6, 0);
          v6 = *(_QWORD *)(a1 + 16);
          v9 = (unsigned int)(8 * v20);
          goto LABEL_6;
        case 0xB:
          v9 = *(_DWORD *)(v8 + 8) >> 8;
          goto LABEL_6;
        case 0xD:
          v13 = (_QWORD *)sub_15A9930(v6, v8);
          v6 = *(_QWORD *)(a1 + 16);
          v9 = 8LL * *v13;
          goto LABEL_6;
        case 0xE:
          v14 = *(_QWORD *)(v8 + 32);
          v41 = *(_QWORD *)(a1 + 16);
          v38 = *(_QWORD *)(v8 + 24);
          v15 = sub_15A9FE0(v6, v38);
          v16 = v38;
          v17 = v41;
          v18 = 1;
          v2 = v15;
          while ( 2 )
          {
            switch ( *(_BYTE *)(v16 + 8) )
            {
              case 0:
              case 8:
              case 0xA:
              case 0xC:
              case 0x10:
                v32 = *(_QWORD *)(v16 + 32);
                v16 = *(_QWORD *)(v16 + 24);
                v18 *= v32;
                continue;
              case 1:
                v26 = 16;
                goto LABEL_33;
              case 2:
                v26 = 32;
                goto LABEL_33;
              case 3:
              case 9:
                v26 = 64;
                goto LABEL_33;
              case 4:
                v26 = 80;
                goto LABEL_33;
              case 5:
              case 6:
                v26 = 128;
                goto LABEL_33;
              case 7:
                v45 = v18;
                v31 = sub_15A9520(v17, 0);
                v18 = v45;
                v26 = (unsigned int)(8 * v31);
                goto LABEL_33;
              case 0xB:
                v26 = *(_DWORD *)(v16 + 8) >> 8;
                goto LABEL_33;
              case 0xD:
                v44 = v18;
                v30 = (_QWORD *)sub_15A9930(v17, v16);
                v18 = v44;
                v26 = 8LL * *v30;
                goto LABEL_33;
              case 0xE:
                v34 = v18;
                v37 = v41;
                v35 = *(_QWORD *)(v16 + 24);
                v43 = *(_QWORD *)(v16 + 32);
                v39 = (unsigned int)sub_15A9FE0(v17, v35);
                v29 = sub_127FA20(v37, v35);
                v18 = v34;
                v26 = 8 * v43 * v39 * ((v39 + ((unsigned __int64)(v29 + 7) >> 3) - 1) / v39);
                goto LABEL_33;
              case 0xF:
                v42 = v18;
                v28 = sub_15A9520(v17, *(_DWORD *)(v16 + 8) >> 8);
                v18 = v42;
                v26 = (unsigned int)(8 * v28);
LABEL_33:
                v6 = *(_QWORD *)(a1 + 16);
                v9 = 8 * v2 * v14 * ((v2 + ((unsigned __int64)(v26 * v18 + 7) >> 3) - 1) / v2);
                break;
            }
            goto LABEL_6;
          }
        case 0xF:
          v19 = sub_15A9520(v6, *(_DWORD *)(v8 + 8) >> 8);
          v6 = *(_QWORD *)(a1 + 16);
          v9 = (unsigned int)(8 * v19);
LABEL_6:
          v10 = 1;
          v11 = (unsigned __int64)(v9 * v7 + 7) >> 3;
          while ( 2 )
          {
            switch ( *(_BYTE *)(v4 + 8) )
            {
              case 1:
                v12 = 16;
                return (unsigned __int64)(v12 * v10 + 7) >> 3 >= v11;
              case 2:
                v12 = 32;
                return (unsigned __int64)(v12 * v10 + 7) >> 3 >= v11;
              case 3:
              case 9:
                v12 = 64;
                return (unsigned __int64)(v12 * v10 + 7) >> 3 >= v11;
              case 4:
                v12 = 80;
                return (unsigned __int64)(v12 * v10 + 7) >> 3 >= v11;
              case 5:
              case 6:
                v12 = 128;
                return (unsigned __int64)(v12 * v10 + 7) >> 3 >= v11;
              case 7:
                v12 = 8 * (unsigned int)sub_15A9520(v6, 0);
                return (unsigned __int64)(v12 * v10 + 7) >> 3 >= v11;
              case 0xB:
                v12 = *(_DWORD *)(v4 + 8) >> 8;
                return (unsigned __int64)(v12 * v10 + 7) >> 3 >= v11;
              case 0xD:
                v12 = 8LL * *(_QWORD *)sub_15A9930(v6, v4);
                return (unsigned __int64)(v12 * v10 + 7) >> 3 >= v11;
              case 0xE:
                v2 = 1;
                v23 = *(_QWORD *)(v4 + 32);
                v24 = *(_QWORD *)(v4 + 24);
                v25 = (unsigned int)sub_15A9FE0(v6, v24);
                while ( 2 )
                {
                  switch ( *(_BYTE *)(v24 + 8) )
                  {
                    case 1:
                      v27 = 16;
                      goto LABEL_36;
                    case 2:
                      v27 = 32;
                      goto LABEL_36;
                    case 3:
                    case 9:
                      v27 = 64;
                      goto LABEL_36;
                    case 4:
                      v27 = 80;
                      goto LABEL_36;
                    case 5:
                    case 6:
                      v27 = 128;
                      goto LABEL_36;
                    case 7:
                      v27 = 8 * (unsigned int)sub_15A9520(v6, 0);
                      goto LABEL_36;
                    case 0xB:
                      v27 = *(_DWORD *)(v24 + 8) >> 8;
                      goto LABEL_36;
                    case 0xD:
                      v27 = 8LL * *(_QWORD *)sub_15A9930(v6, v24);
                      goto LABEL_36;
                    case 0xE:
                      v36 = *(_QWORD *)(v24 + 24);
                      v46 = *(_QWORD *)(v24 + 32);
                      v40 = (unsigned int)sub_15A9FE0(v6, v36);
                      v27 = 8 * v40 * v46 * ((v40 + ((unsigned __int64)(sub_127FA20(v6, v36) + 7) >> 3) - 1) / v40);
                      goto LABEL_36;
                    case 0xF:
                      v27 = 8 * (unsigned int)sub_15A9520(v6, *(_DWORD *)(v24 + 8) >> 8);
LABEL_36:
                      v12 = 8 * v25 * v23 * ((v25 + ((v27 * v2 + 7) >> 3) - 1) / v25);
                      return (unsigned __int64)(v12 * v10 + 7) >> 3 >= v11;
                    case 0x10:
                      v33 = *(_QWORD *)(v24 + 32);
                      v24 = *(_QWORD *)(v24 + 24);
                      v2 *= v33;
                      continue;
                    default:
                      goto LABEL_4;
                  }
                }
              case 0xF:
                v12 = 8 * (unsigned int)sub_15A9520(v6, *(_DWORD *)(v4 + 8) >> 8);
                return (unsigned __int64)(v12 * v10 + 7) >> 3 >= v11;
              case 0x10:
                v22 = *(_QWORD *)(v4 + 32);
                v4 = *(_QWORD *)(v4 + 24);
                v10 *= v22;
                continue;
              default:
LABEL_4:
                ++*(_DWORD *)(v2 + 16);
                BUG();
            }
          }
      }
    }
  }
  return result;
}
