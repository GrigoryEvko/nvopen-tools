// Function: sub_15A9C40
// Address: 0x15a9c40
//
__int64 __fastcall sub_15A9C40(__int64 a1, __int64 a2, unsigned __int8 a3)
{
  __int64 v4; // rbx
  char v5; // si
  unsigned int v6; // r14d
  __int64 v7; // r8
  __int64 v8; // r13
  __int64 result; // rax
  __int64 v10; // r14
  unsigned int v11; // esi
  __int64 v12; // rdx
  __int64 v13; // r15
  unsigned int v14; // eax
  __int64 v15; // r9
  unsigned __int64 v16; // r8
  __int64 v17; // rax
  __int64 v18; // rax
  unsigned int v19; // esi
  int v20; // eax
  __int64 v21; // rax
  unsigned __int64 v22; // r15
  __int64 v23; // rax
  _QWORD *v24; // rax
  __int64 v25; // [rsp+0h] [rbp-60h]
  unsigned __int64 v26; // [rsp+8h] [rbp-58h]
  __int64 v27; // [rsp+10h] [rbp-50h]
  __int64 v28; // [rsp+10h] [rbp-50h]
  __int64 v29; // [rsp+10h] [rbp-50h]
  unsigned __int64 v30; // [rsp+18h] [rbp-48h]
  unsigned __int64 v31; // [rsp+18h] [rbp-48h]
  __int64 v32; // [rsp+20h] [rbp-40h]
  __int64 v33; // [rsp+28h] [rbp-38h]

  while ( 2 )
  {
    v4 = a2;
    v5 = *(_BYTE *)(a2 + 8);
    switch ( v5 )
    {
      case 0:
      case 8:
      case 9:
      case 10:
      case 12:
      case 16:
        v6 = 118;
        goto LABEL_4;
      case 1:
      case 2:
      case 3:
      case 4:
      case 5:
      case 6:
        v6 = 102;
        goto LABEL_4;
      case 7:
        v11 = 0;
        if ( !a3 )
          return sub_15A94D0(a1, v11);
        return sub_15A9480(a1, v11);
      case 11:
        v6 = 105;
LABEL_4:
        v7 = v4;
        v8 = 1;
        while ( 2 )
        {
          switch ( v5 )
          {
            case 1:
              v12 = 16;
              return sub_15AA980(a1, v6, v8 * v12, a3, v4);
            case 2:
              v12 = 32;
              return sub_15AA980(a1, v6, v8 * v12, a3, v4);
            case 3:
            case 9:
              v12 = 64;
              return sub_15AA980(a1, v6, v8 * v12, a3, v4);
            case 4:
              v12 = 80;
              return sub_15AA980(a1, v6, v8 * v12, a3, v4);
            case 5:
            case 6:
              v12 = 128;
              return sub_15AA980(a1, v6, v8 * v12, a3, v4);
            case 7:
              v12 = 8 * (unsigned int)sub_15A9520(a1, 0);
              return sub_15AA980(a1, v6, v8 * v12, a3, v4);
            case 11:
              v12 = *(_DWORD *)(v7 + 8) >> 8;
              return sub_15AA980(a1, v6, v8 * v12, a3, v4);
            case 13:
              v12 = 8LL * *(_QWORD *)sub_15A9930(a1, v7);
              return sub_15AA980(a1, v6, v8 * v12, a3, v4);
            case 14:
              v13 = *(_QWORD *)(v7 + 24);
              v33 = *(_QWORD *)(v7 + 32);
              v14 = sub_15A9FE0(a1, v13);
              v15 = 1;
              v16 = v14;
              while ( 2 )
              {
                switch ( *(_BYTE *)(v13 + 8) )
                {
                  case 1:
                    v18 = 16;
                    goto LABEL_32;
                  case 2:
                    v18 = 32;
                    goto LABEL_32;
                  case 3:
                  case 9:
                    v18 = 64;
                    goto LABEL_32;
                  case 4:
                    v18 = 80;
                    goto LABEL_32;
                  case 5:
                  case 6:
                    v18 = 128;
                    goto LABEL_32;
                  case 7:
                    v27 = v15;
                    v19 = 0;
                    v30 = v16;
                    goto LABEL_36;
                  case 0xB:
                    v18 = *(_DWORD *)(v13 + 8) >> 8;
                    goto LABEL_32;
                  case 0xD:
                    v29 = v15;
                    v31 = v16;
                    v24 = (_QWORD *)sub_15A9930(a1, v13);
                    v16 = v31;
                    v15 = v29;
                    v18 = 8LL * *v24;
                    goto LABEL_32;
                  case 0xE:
                    v25 = v15;
                    v26 = v16;
                    v28 = *(_QWORD *)(v13 + 24);
                    v32 = *(_QWORD *)(v13 + 32);
                    v22 = (unsigned int)sub_15A9FE0(a1, v28);
                    v23 = sub_127FA20(a1, v28);
                    v16 = v26;
                    v15 = v25;
                    v18 = 8 * v22 * v32 * ((v22 + ((unsigned __int64)(v23 + 7) >> 3) - 1) / v22);
                    goto LABEL_32;
                  case 0xF:
                    v27 = v15;
                    v30 = v16;
                    v19 = *(_DWORD *)(v13 + 8) >> 8;
LABEL_36:
                    v20 = sub_15A9520(a1, v19);
                    v16 = v30;
                    v15 = v27;
                    v18 = (unsigned int)(8 * v20);
LABEL_32:
                    v12 = 8 * v16 * v33 * ((v16 + ((unsigned __int64)(v18 * v15 + 7) >> 3) - 1) / v16);
                    return sub_15AA980(a1, v6, v8 * v12, a3, v4);
                  case 0x10:
                    v21 = *(_QWORD *)(v13 + 32);
                    v13 = *(_QWORD *)(v13 + 24);
                    v15 *= v21;
                    continue;
                  default:
                    goto LABEL_2;
                }
              }
            case 15:
              v12 = 8 * (unsigned int)sub_15A9520(a1, *(_DWORD *)(v7 + 8) >> 8);
              return sub_15AA980(a1, v6, v8 * v12, a3, v4);
            case 16:
              v17 = *(_QWORD *)(v7 + 32);
              v7 = *(_QWORD *)(v7 + 24);
              v8 *= v17;
              v5 = *(_BYTE *)(v7 + 8);
              continue;
            default:
LABEL_2:
              BUG();
          }
        }
      case 13:
        if ( (*(_BYTE *)(v4 + 9) & 2) == 0 || (result = 1, !a3) )
        {
          v10 = sub_15A9930(a1, v4);
          result = sub_15AA980(a1, 97, 0, a3, v4);
          if ( *(_DWORD *)(v10 + 8) >= (unsigned int)result )
            return *(unsigned int *)(v10 + 8);
        }
        return result;
      case 14:
        a2 = *(_QWORD *)(v4 + 24);
        continue;
      case 15:
        v11 = *(_DWORD *)(v4 + 8) >> 8;
        if ( a3 )
          return sub_15A9480(a1, v11);
        else
          return sub_15A94D0(a1, v11);
    }
  }
}
