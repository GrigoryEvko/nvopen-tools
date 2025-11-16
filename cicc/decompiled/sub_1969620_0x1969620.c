// Function: sub_1969620
// Address: 0x1969620
//
__int64 __fastcall sub_1969620(__int64 a1, _BYTE *a2)
{
  __int64 v3; // r12
  __int64 *v4; // rsi
  __int64 result; // rax
  __int64 v6; // rax
  unsigned __int64 v7; // rax
  __int64 v8; // r12
  unsigned __int64 v9; // r14
  __int64 *v10; // rax
  __int64 v11; // rcx
  signed __int64 v12; // r13
  __int64 *v13; // r15
  __int64 v14; // rcx
  __int64 v15; // r14
  _QWORD *v16; // rdx
  __int128 v17; // rdi
  __int64 v18; // rax
  __int64 v19; // r14
  unsigned int v20; // eax
  __int64 v21; // rsi
  __int64 v22; // rcx
  unsigned __int64 v23; // r15
  __int64 v24; // rax
  __int64 v25; // rax
  int v26; // eax
  int v27; // eax
  __int64 v28; // rax
  _QWORD *v29; // rax
  __int64 v30; // [rsp-58h] [rbp-58h]
  __int64 v31; // [rsp-50h] [rbp-50h]
  unsigned __int64 v32; // [rsp-48h] [rbp-48h]
  __int64 v33; // [rsp-40h] [rbp-40h]
  __int64 v34; // [rsp-40h] [rbp-40h]
  __int64 v35; // [rsp-40h] [rbp-40h]
  __int64 v36; // [rsp-40h] [rbp-40h]
  __int64 v37; // [rsp-40h] [rbp-40h]
  __int64 v38; // [rsp-40h] [rbp-40h]

  if ( *(_BYTE *)(a1 + 16) > 0x10u )
    return 0;
  v3 = 1;
  v4 = *(__int64 **)a1;
  while ( 2 )
  {
    switch ( *((_BYTE *)v4 + 8) )
    {
      case 0:
      case 8:
      case 0xA:
      case 0xC:
      case 0x10:
        v18 = v4[4];
        v4 = (__int64 *)v4[3];
        v3 *= v18;
        continue;
      case 1:
        v6 = 16;
        break;
      case 2:
        v6 = 32;
        break;
      case 3:
      case 9:
        v6 = 64;
        break;
      case 4:
        v6 = 80;
        break;
      case 5:
      case 6:
        v6 = 128;
        break;
      case 7:
        v6 = 8 * (unsigned int)sub_15A9520((__int64)a2, 0);
        break;
      case 0xB:
        v6 = *((_DWORD *)v4 + 2) >> 8;
        break;
      case 0xD:
        v6 = 8LL * *(_QWORD *)sub_15A9930((__int64)a2, (__int64)v4);
        break;
      case 0xE:
        v19 = v4[4];
        v34 = v4[3];
        v20 = sub_15A9FE0((__int64)a2, v34);
        v21 = v34;
        v22 = 1;
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
              v25 = *(_QWORD *)(v21 + 32);
              v21 = *(_QWORD *)(v21 + 24);
              v22 *= v25;
              continue;
            case 1:
              v24 = 16;
              break;
            case 2:
              v24 = 32;
              break;
            case 3:
            case 9:
              v24 = 64;
              break;
            case 4:
              v24 = 80;
              break;
            case 5:
            case 6:
              v24 = 128;
              break;
            case 7:
              v35 = v22;
              v26 = sub_15A9520((__int64)a2, 0);
              v22 = v35;
              v24 = (unsigned int)(8 * v26);
              break;
            case 0xB:
              v24 = *(_DWORD *)(v21 + 8) >> 8;
              break;
            case 0xD:
              v38 = v22;
              v29 = (_QWORD *)sub_15A9930((__int64)a2, v21);
              v22 = v38;
              v24 = 8LL * *v29;
              break;
            case 0xE:
              v30 = v22;
              v31 = *(_QWORD *)(v21 + 24);
              v37 = *(_QWORD *)(v21 + 32);
              v32 = (unsigned int)sub_15A9FE0((__int64)a2, v31);
              v28 = sub_127FA20((__int64)a2, v31);
              v22 = v30;
              v24 = 8 * v32 * v37 * ((v32 + ((unsigned __int64)(v28 + 7) >> 3) - 1) / v32);
              break;
            case 0xF:
              v36 = v22;
              v27 = sub_15A9520((__int64)a2, *(_DWORD *)(v21 + 8) >> 8);
              v22 = v36;
              v24 = (unsigned int)(8 * v27);
              break;
          }
          break;
        }
        v6 = 8 * v23 * v19 * ((v23 + ((unsigned __int64)(v24 * v22 + 7) >> 3) - 1) / v23);
        break;
      case 0xF:
        v6 = 8 * (unsigned int)sub_15A9520((__int64)a2, *((_DWORD *)v4 + 2) >> 8);
        break;
    }
    break;
  }
  v7 = v3 * v6;
  if ( !v7 )
    return 0;
  v8 = v7 & ((v7 - 1) | 7);
  if ( v8 )
    return 0;
  if ( *a2 )
    return 0;
  v9 = v7 >> 3;
  if ( v7 > 0x87 )
    return 0;
  if ( v9 == 16 )
    return a1;
  v10 = sub_1645D80(*(__int64 **)a1, 0x10 / v9);
  v12 = 8 * (0x10 / v9);
  v13 = v10;
  if ( v9 > 0x10 )
    return sub_159DFD0((unsigned __int64)v10, 0, v11);
  v15 = sub_22077B0(v12);
  if ( v15 == v12 + v15 )
  {
    v12 = 0;
  }
  else
  {
    v16 = (_QWORD *)v15;
    do
      *v16++ = a1;
    while ( v16 != (_QWORD *)(v12 + v15) );
    v8 = v12 >> 3;
  }
  *((_QWORD *)&v17 + 1) = v15;
  *(_QWORD *)&v17 = v13;
  result = sub_159DFD0(v17, v8, v14);
  if ( v15 )
  {
    v33 = result;
    j_j___libc_free_0(v15, v12);
    return v33;
  }
  return result;
}
