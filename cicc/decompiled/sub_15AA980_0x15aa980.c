// Function: sub_15AA980
// Address: 0x15aa980
//
__int64 __fastcall sub_15AA980(__int64 a1, unsigned int a2, unsigned int a3, char a4, __int64 a5)
{
  _DWORD *v8; // rax
  _DWORD *v9; // rdx
  __int64 v10; // rbx
  unsigned int v11; // r8d
  __int64 v13; // rax
  __int64 v14; // rax
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // rax
  __int64 v17; // r12
  __int64 v18; // r14
  __int64 v19; // rsi
  unsigned __int64 v20; // r15
  __int64 v21; // rax
  __int64 v22; // r14
  __int64 v23; // rbx
  unsigned __int64 v24; // r12
  __int64 v25; // rax
  unsigned __int64 v26; // rtt
  __int64 v27; // rax
  __int64 v28; // rax
  unsigned __int64 v29; // r14
  __int64 v30; // rax
  __int64 v31; // [rsp+8h] [rbp-48h]
  __int64 v32; // [rsp+10h] [rbp-40h]
  unsigned __int64 v33; // [rsp+10h] [rbp-40h]
  __int64 v34; // [rsp+18h] [rbp-38h]
  __int64 v35; // [rsp+18h] [rbp-38h]

  v8 = sub_15A8270(a1, a2, a3);
  v9 = *(_DWORD **)(a1 + 48);
  if ( v8 != &v9[2 * *(unsigned int *)(a1 + 56)] && *(unsigned __int8 *)v8 == a2 )
  {
    if ( *v8 >> 8 == a3 || a2 == 105 )
    {
      if ( a4 )
        return *((unsigned __int16 *)v8 + 2);
      else
        return *((unsigned __int16 *)v8 + 3);
    }
  }
  else if ( a2 == 105 )
  {
    v10 = 1;
    if ( v8 == v9 || *((_BYTE *)v8 - 8) != 105 )
      goto LABEL_6;
    if ( a4 )
      return *((unsigned __int16 *)v8 - 2);
    else
      return *((unsigned __int16 *)v8 - 1);
  }
  if ( a2 == 118 )
  {
    v22 = *(_QWORD *)(a5 + 24);
    v23 = 1;
    v24 = (unsigned int)sub_15A9FE0(a1, v22);
    while ( 2 )
    {
      switch ( *(_BYTE *)(v22 + 8) )
      {
        case 0:
        case 8:
        case 0xA:
        case 0xC:
        case 0x10:
          v28 = *(_QWORD *)(v22 + 32);
          v22 = *(_QWORD *)(v22 + 24);
          v23 *= v28;
          continue;
        case 1:
          v25 = 16;
          break;
        case 2:
          v25 = 32;
          break;
        case 3:
        case 9:
          v25 = 64;
          break;
        case 4:
          v25 = 80;
          break;
        case 5:
        case 6:
          v25 = 128;
          break;
        case 7:
          v25 = 8 * (unsigned int)sub_15A9520(a1, 0);
          break;
        case 0xB:
          v25 = *(_DWORD *)(v22 + 8) >> 8;
          break;
        case 0xD:
          v25 = 8LL * *(_QWORD *)sub_15A9930(a1, v22);
          break;
        case 0xE:
          v32 = *(_QWORD *)(v22 + 24);
          v34 = *(_QWORD *)(v22 + 32);
          v29 = (unsigned int)sub_15A9FE0(a1, v32);
          v25 = 8 * v34 * v29 * ((v29 + ((unsigned __int64)(sub_127FA20(a1, v32) + 7) >> 3) - 1) / v29);
          break;
        case 0xF:
          v25 = 8 * (unsigned int)sub_15A9520(a1, *(_DWORD *)(v22 + 8) >> 8);
          break;
      }
      break;
    }
    v11 = 0;
    v26 = v24 + ((unsigned __int64)(v25 * v23 + 7) >> 3) - 1;
    v14 = *(_DWORD *)(a5 + 32) * (unsigned int)v24 * (unsigned int)(v26 / v24);
    if ( *(_DWORD *)(a5 + 32) * (_DWORD)v24 * (unsigned int)(v26 / v24) )
    {
LABEL_19:
      v15 = ((((unsigned __int64)(v14 - 1) >> 1) | (v14 - 1)) >> 2) | ((unsigned __int64)(v14 - 1) >> 1) | (v14 - 1);
      v16 = (((v15 >> 4) | v15) >> 8) | (v15 >> 4) | v15;
      return ((unsigned int)(v16 >> 16) | (unsigned int)v16) + 1;
    }
  }
  else
  {
    v10 = 1;
    if ( a2 != 97 )
    {
LABEL_6:
      while ( 2 )
      {
        switch ( *(_BYTE *)(a5 + 8) )
        {
          case 1:
            v13 = 16;
            goto LABEL_18;
          case 2:
            v13 = 32;
            goto LABEL_18;
          case 3:
          case 9:
            v13 = 64;
            goto LABEL_18;
          case 4:
            v13 = 80;
            goto LABEL_18;
          case 5:
          case 6:
            v13 = 128;
            goto LABEL_18;
          case 7:
            v13 = 8 * (unsigned int)sub_15A9520(a1, 0);
            goto LABEL_18;
          case 0xB:
            v13 = *(_DWORD *)(a5 + 8) >> 8;
            goto LABEL_18;
          case 0xD:
            v13 = 8LL * *(_QWORD *)sub_15A9930(a1, a5);
            goto LABEL_18;
          case 0xE:
            v17 = *(_QWORD *)(a5 + 32);
            v18 = 1;
            v19 = *(_QWORD *)(a5 + 24);
            v20 = (unsigned int)sub_15A9FE0(a1, v19);
            while ( 2 )
            {
              switch ( *(_BYTE *)(v19 + 8) )
              {
                case 1:
                  v27 = 16;
                  goto LABEL_39;
                case 2:
                  v27 = 32;
                  goto LABEL_39;
                case 3:
                case 9:
                  v27 = 64;
                  goto LABEL_39;
                case 4:
                  v27 = 80;
                  goto LABEL_39;
                case 5:
                case 6:
                  v27 = 128;
                  goto LABEL_39;
                case 7:
                  v27 = 8 * (unsigned int)sub_15A9520(a1, 0);
                  goto LABEL_39;
                case 0xB:
                  v27 = *(_DWORD *)(v19 + 8) >> 8;
                  goto LABEL_39;
                case 0xD:
                  v27 = 8LL * *(_QWORD *)sub_15A9930(a1, v19);
                  goto LABEL_39;
                case 0xE:
                  v31 = *(_QWORD *)(v19 + 24);
                  v35 = *(_QWORD *)(v19 + 32);
                  v33 = (unsigned int)sub_15A9FE0(a1, v31);
                  v27 = 8 * v33 * v35 * ((v33 + ((unsigned __int64)(sub_127FA20(a1, v31) + 7) >> 3) - 1) / v33);
                  goto LABEL_39;
                case 0xF:
                  v27 = 8 * (unsigned int)sub_15A9520(a1, *(_DWORD *)(v19 + 8) >> 8);
LABEL_39:
                  v13 = 8 * v20 * v17 * ((v20 + ((unsigned __int64)(v27 * v18 + 7) >> 3) - 1) / v20);
                  goto LABEL_18;
                case 0x10:
                  v30 = *(_QWORD *)(v19 + 32);
                  v19 = *(_QWORD *)(v19 + 24);
                  v18 *= v30;
                  continue;
                default:
                  goto LABEL_60;
              }
            }
          case 0xF:
            v13 = 8 * (unsigned int)sub_15A9520(a1, *(_DWORD *)(a5 + 8) >> 8);
LABEL_18:
            v11 = 0;
            v14 = (unsigned int)((unsigned __int64)(v13 * v10 + 7) >> 3);
            if ( (_DWORD)v14 )
              goto LABEL_19;
            return v11;
          case 0x10:
            v21 = *(_QWORD *)(a5 + 32);
            a5 = *(_QWORD *)(a5 + 24);
            v10 *= v21;
            continue;
          default:
LABEL_60:
            BUG();
        }
      }
    }
    return *(unsigned int *)(sub_15A9930(a1, a5) + 8);
  }
  return v11;
}
