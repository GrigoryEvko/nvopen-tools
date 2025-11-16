// Function: sub_15AA470
// Address: 0x15aa470
//
unsigned __int64 __fastcall sub_15AA470(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // rdx
  unsigned __int64 result; // rax
  __int64 v7; // r13
  __int64 v8; // rax
  __int64 v9; // r12
  unsigned __int64 v10; // rax
  unsigned int v11; // ecx
  unsigned int v12; // eax
  __int64 v13; // rcx
  unsigned __int64 v14; // r15
  __int64 v15; // rax
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // r13
  int v18; // eax
  __int64 v19; // rax
  int v20; // eax
  unsigned int v21; // eax
  __int64 v22; // rsi
  __int64 v23; // r10
  unsigned __int64 v24; // r12
  _QWORD *v25; // rax
  __int64 v26; // rax
  _QWORD *v27; // rax
  __int64 v28; // rsi
  __int64 v29; // r9
  unsigned int v30; // esi
  int v31; // eax
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // [rsp+38h] [rbp-58h]
  __int64 v35; // [rsp+38h] [rbp-58h]
  __int64 v36; // [rsp+40h] [rbp-50h]
  __int64 v37; // [rsp+40h] [rbp-50h]
  __int64 v38; // [rsp+40h] [rbp-50h]
  __int64 v39; // [rsp+48h] [rbp-48h]
  __int64 v40; // [rsp+48h] [rbp-48h]
  __int64 v41; // [rsp+48h] [rbp-48h]
  __int64 v42; // [rsp+48h] [rbp-48h]
  __int64 v44; // [rsp+58h] [rbp-38h]

  v4 = *(unsigned int *)(a2 + 12);
  *(_QWORD *)a1 = 0;
  v5 = v4 << 33;
  result = v4 & 0x7FFFFFFF;
  *(_QWORD *)(a1 + 8) = v5;
  v44 = 8LL * (unsigned int)result;
  if ( !(_DWORD)result )
    goto LABEL_28;
  v7 = 0;
  do
  {
    v8 = *(_QWORD *)(a2 + 16);
    v9 = *(_QWORD *)(v8 + v7);
    if ( (*(_BYTE *)(a2 + 9) & 2) != 0 )
    {
      v10 = *(_QWORD *)a1;
      v11 = 1;
    }
    else
    {
      v11 = sub_15A9FE0(a3, *(_QWORD *)(v8 + v7));
      v10 = *(_QWORD *)a1;
      if ( (*(_QWORD *)a1 & (v11 - 1)) != 0 )
      {
        *(_BYTE *)(a1 + 12) |= 1u;
        v10 = v11 * ((v11 + v10 - 1) / v11);
        *(_QWORD *)a1 = v10;
      }
    }
    if ( *(_DWORD *)(a1 + 8) >= v11 )
      v11 = *(_DWORD *)(a1 + 8);
    *(_DWORD *)(a1 + 8) = v11;
    *(_QWORD *)(a1 + v7 + 16) = v10;
    v12 = sub_15A9FE0(a3, v9);
    v13 = 1;
    v14 = v12;
    while ( 2 )
    {
      switch ( *(_BYTE *)(v9 + 8) )
      {
        case 0:
        case 8:
        case 0xA:
        case 0xC:
        case 0x10:
          v19 = *(_QWORD *)(v9 + 32);
          v9 = *(_QWORD *)(v9 + 24);
          v13 *= v19;
          continue;
        case 1:
          v15 = 16;
          break;
        case 2:
          v15 = 32;
          break;
        case 3:
        case 9:
          v15 = 64;
          break;
        case 4:
          v15 = 80;
          break;
        case 5:
        case 6:
          v15 = 128;
          break;
        case 7:
          v39 = v13;
          v18 = sub_15A9520(a3, 0);
          v13 = v39;
          v15 = (unsigned int)(8 * v18);
          break;
        case 0xB:
          v15 = *(_DWORD *)(v9 + 8) >> 8;
          break;
        case 0xD:
          v42 = v13;
          v25 = (_QWORD *)sub_15A9930(a3, v9);
          v13 = v42;
          v15 = 8LL * *v25;
          break;
        case 0xE:
          v34 = v13;
          v36 = *(_QWORD *)(v9 + 24);
          v41 = *(_QWORD *)(v9 + 32);
          v21 = sub_15A9FE0(a3, v36);
          v22 = v36;
          v13 = v34;
          v23 = 1;
          v24 = v21;
          while ( 2 )
          {
            switch ( *(_BYTE *)(v22 + 8) )
            {
              case 0:
              case 8:
              case 0xA:
              case 0xC:
              case 0x10:
                v32 = *(_QWORD *)(v22 + 32);
                v22 = *(_QWORD *)(v22 + 24);
                v23 *= v32;
                continue;
              case 1:
                v26 = 16;
                goto LABEL_30;
              case 2:
                v26 = 32;
                goto LABEL_30;
              case 3:
              case 9:
                v26 = 64;
                goto LABEL_30;
              case 4:
                v26 = 80;
                goto LABEL_30;
              case 5:
              case 6:
                v26 = 128;
                goto LABEL_30;
              case 7:
                v30 = 0;
                v38 = v23;
                goto LABEL_39;
              case 0xB:
                v26 = *(_DWORD *)(v22 + 8) >> 8;
                goto LABEL_30;
              case 0xD:
                v37 = v23;
                v27 = (_QWORD *)sub_15A9930(a3, v22);
                v23 = v37;
                v13 = v34;
                v26 = 8LL * *v27;
                goto LABEL_30;
              case 0xE:
                v35 = *(_QWORD *)(v22 + 24);
                sub_15A9FE0(a3, v35);
                v28 = v35;
                v29 = 1;
                while ( 1 )
                {
                  switch ( *(_BYTE *)(v28 + 8) )
                  {
                    case 0:
                      v33 = *(_QWORD *)(v28 + 32);
                      v28 = *(_QWORD *)(v28 + 24);
                      v29 *= v33;
                      break;
                    case 1:
                    case 2:
                    case 5:
                    case 6:
                      JUMPOUT(0x15AA821);
                    case 3:
                      JUMPOUT(0x15AA81C);
                    case 4:
                      JUMPOUT(0x15AA853);
                  }
                }
              case 0xF:
                v38 = v23;
                v30 = *(_DWORD *)(v22 + 8) >> 8;
LABEL_39:
                v31 = sub_15A9520(a3, v30);
                v23 = v38;
                v13 = v34;
                v26 = (unsigned int)(8 * v31);
LABEL_30:
                v15 = 8 * v24 * v41 * ((v24 + ((unsigned __int64)(v26 * v23 + 7) >> 3) - 1) / v24);
                break;
            }
            break;
          }
          break;
        case 0xF:
          v40 = v13;
          v20 = sub_15A9520(a3, *(_DWORD *)(v9 + 8) >> 8);
          v13 = v40;
          v15 = (unsigned int)(8 * v20);
          break;
      }
      break;
    }
    v7 += 8;
    v16 = *(_QWORD *)a1 + v14 * ((v14 + ((unsigned __int64)(v15 * v13 + 7) >> 3) - 1) / v14);
    *(_QWORD *)a1 = v16;
  }
  while ( v44 != v7 );
  v17 = v16;
  result = *(unsigned int *)(a1 + 8);
  if ( (_DWORD)result )
  {
    if ( ((unsigned int)v17 & ((_DWORD)result - 1)) != 0 )
    {
      *(_BYTE *)(a1 + 12) |= 1u;
      result = (unsigned int)result * ((v17 + (unsigned int)result - 1) / (unsigned int)result);
      *(_QWORD *)a1 = result;
    }
  }
  else
  {
LABEL_28:
    *(_DWORD *)(a1 + 8) = 1;
  }
  return result;
}
