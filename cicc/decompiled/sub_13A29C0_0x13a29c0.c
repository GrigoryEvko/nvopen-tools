// Function: sub_13A29C0
// Address: 0x13a29c0
//
__int64 __fastcall sub_13A29C0(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v4; // rax
  __int64 v5; // r14
  __int64 v6; // rax
  __int64 v7; // rcx
  unsigned int v8; // edx
  __int64 v9; // rsi
  __int64 *v10; // r8
  unsigned int v11; // eax
  int v13; // esi
  __int64 v14; // rsi
  int v15; // ebx
  __int64 v16; // rcx
  unsigned int v17; // ecx
  __int64 v18; // r13
  __int64 v19; // r15
  unsigned __int64 v20; // rcx
  __int64 v21; // rax
  int v22; // r10d
  __int64 v23; // rax
  unsigned __int64 v24; // r13
  __int64 v25; // rax
  _QWORD *v26; // rax
  int v27; // eax
  __int64 v28; // rax
  unsigned __int64 v29; // [rsp+0h] [rbp-50h]
  __int64 v30; // [rsp+8h] [rbp-48h]
  __int64 v31; // [rsp+10h] [rbp-40h]
  unsigned __int64 v32; // [rsp+10h] [rbp-40h]
  unsigned __int64 v33; // [rsp+10h] [rbp-40h]
  __int64 v34; // [rsp+18h] [rbp-38h]

  sub_13A29A0(a2);
  v4 = sub_15F2050(a3);
  v5 = sub_1632FA0(v4);
  v6 = *(unsigned int *)(a2 + 352);
  if ( (_DWORD)v6 )
  {
    v7 = *(_QWORD *)(a2 + 336);
    v8 = (v6 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
    v9 = v7 + 24LL * v8;
    v10 = *(__int64 **)v9;
    if ( a3 == *(__int64 **)v9 )
    {
LABEL_3:
      if ( v9 != v7 + 24 * v6 )
      {
        v11 = *(_DWORD *)(v9 + 16);
        *(_DWORD *)(a1 + 8) = v11;
        if ( v11 > 0x40 )
          sub_16A4FD0(a1, v9 + 8);
        else
          *(_QWORD *)a1 = *(_QWORD *)(v9 + 8);
        return a1;
      }
    }
    else
    {
      v13 = 1;
      while ( v10 != (__int64 *)-8LL )
      {
        v22 = v13 + 1;
        v8 = (v6 - 1) & (v13 + v8);
        v9 = v7 + 24LL * v8;
        v10 = *(__int64 **)v9;
        if ( a3 == *(__int64 **)v9 )
          goto LABEL_3;
        v13 = v22;
      }
    }
  }
  v14 = *a3;
  v15 = 1;
  while ( 2 )
  {
    switch ( *(_BYTE *)(v14 + 8) )
    {
      case 0:
      case 8:
      case 0xA:
      case 0xC:
      case 0x10:
        v21 = *(_QWORD *)(v14 + 32);
        v14 = *(_QWORD *)(v14 + 24);
        v15 *= (_DWORD)v21;
        continue;
      case 1:
        LODWORD(v16) = 16;
        break;
      case 2:
        LODWORD(v16) = 32;
        break;
      case 3:
      case 9:
        LODWORD(v16) = 64;
        break;
      case 4:
        LODWORD(v16) = 80;
        break;
      case 5:
      case 6:
        LODWORD(v16) = 128;
        break;
      case 7:
        LODWORD(v16) = 8 * sub_15A9520(v5, 0);
        break;
      case 0xB:
        LODWORD(v16) = *(_DWORD *)(v14 + 8) >> 8;
        break;
      case 0xD:
        v16 = 8LL * *(_QWORD *)sub_15A9930(v5, v14);
        break;
      case 0xE:
        v18 = *(_QWORD *)(v14 + 24);
        v19 = 1;
        v34 = *(_QWORD *)(v14 + 32);
        v20 = (unsigned int)sub_15A9FE0(v5, v18);
        while ( 2 )
        {
          switch ( *(_BYTE *)(v18 + 8) )
          {
            case 0:
            case 8:
            case 0xA:
            case 0xC:
            case 0x10:
              v28 = *(_QWORD *)(v18 + 32);
              v18 = *(_QWORD *)(v18 + 24);
              v19 *= v28;
              continue;
            case 1:
              v23 = 16;
              goto LABEL_30;
            case 2:
              v23 = 32;
              goto LABEL_30;
            case 3:
            case 9:
              v23 = 64;
              goto LABEL_30;
            case 4:
              v23 = 80;
              goto LABEL_30;
            case 5:
            case 6:
              v23 = 128;
              goto LABEL_30;
            case 7:
              sub_15A9520(v5, 0);
              JUMPOUT(0x13A2CE9);
            case 0xB:
              JUMPOUT(0x13A2CCF);
            case 0xD:
              v32 = v20;
              v26 = (_QWORD *)sub_15A9930(v5, v18);
              v20 = v32;
              v23 = 8LL * *v26;
              goto LABEL_30;
            case 0xE:
              v29 = v20;
              v30 = *(_QWORD *)(v18 + 24);
              v31 = *(_QWORD *)(v18 + 32);
              v24 = (unsigned int)sub_15A9FE0(v5, v30);
              v25 = sub_127FA20(v5, v30);
              v20 = v29;
              v23 = 8 * v24 * v31 * ((v24 + ((unsigned __int64)(v25 + 7) >> 3) - 1) / v24);
              goto LABEL_30;
            case 0xF:
              v33 = v20;
              v27 = sub_15A9520(v5, *(_DWORD *)(v18 + 8) >> 8);
              v20 = v33;
              v23 = (unsigned int)(8 * v27);
LABEL_30:
              v16 = 8 * v34 * v20 * ((v20 + ((unsigned __int64)(v23 * v19 + 7) >> 3) - 1) / v20);
              break;
          }
          break;
        }
        break;
      case 0xF:
        LODWORD(v16) = 8 * sub_15A9520(v5, *(_DWORD *)(v14 + 8) >> 8);
        break;
    }
    break;
  }
  v17 = v15 * v16;
  *(_DWORD *)(a1 + 8) = v17;
  if ( v17 > 0x40 )
    sub_16A4EF0(a1, -1, 1);
  else
    *(_QWORD *)a1 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v17;
  return a1;
}
