// Function: sub_15E2EC0
// Address: 0x15e2ec0
//
char __fastcall sub_15E2EC0(__int64 a1, unsigned int **a2, __int64 *a3)
{
  unsigned int *v3; // rax
  unsigned int *v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r13
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r13
  __int64 v14; // r13
  __int64 v15; // r13
  __int64 v16; // rcx
  __int64 v17; // r13
  __int64 v18; // rax
  __int64 v19; // r13
  _QWORD *v20; // rbx
  char v21; // al
  __int64 v22; // rsi
  __int64 v23; // rax
  __int64 v24; // r13
  char v25; // al
  __int64 v26; // rdx
  __int64 v27; // r15
  __int64 v28; // rax

  v3 = a2[1];
  if ( v3 )
  {
    v6 = *a2;
    while ( 2 )
    {
      v7 = *v6;
      v8 = v6[1];
      v3 = (unsigned int *)((char *)v3 - 1);
      v6 += 2;
      *a2 = v6;
      a2[1] = v3;
      switch ( v7 )
      {
        case 0LL:
          LOBYTE(v9) = *(_BYTE *)(a1 + 8) != 0;
          return v9;
        case 1LL:
          goto LABEL_6;
        case 2LL:
          LOBYTE(v9) = *(_BYTE *)(a1 + 8) != 9;
          return v9;
        case 3LL:
          LOBYTE(v9) = *(_BYTE *)(a1 + 8) != 10;
          return v9;
        case 4LL:
          LOBYTE(v9) = *(_BYTE *)(a1 + 8) != 8;
          return v9;
        case 5LL:
          LOBYTE(v9) = *(_BYTE *)(a1 + 8) != 1;
          return v9;
        case 6LL:
          LOBYTE(v9) = *(_BYTE *)(a1 + 8) != 2;
          return v9;
        case 7LL:
          LOBYTE(v9) = *(_BYTE *)(a1 + 8) != 3;
          return v9;
        case 8LL:
          LOBYTE(v9) = *(_BYTE *)(a1 + 8) != 5;
          return v9;
        case 9LL:
          LODWORD(v9) = sub_1642F90(a1, (unsigned int)v8) ^ 1;
          return v9;
        case 10LL:
          if ( *(_BYTE *)(a1 + 8) != 16 || v8 != *(_QWORD *)(a1 + 32) )
            goto LABEL_6;
          goto LABEL_47;
        case 11LL:
          if ( *(_BYTE *)(a1 + 8) != 15 || *(_DWORD *)(a1 + 8) >> 8 != (_DWORD)v8 )
            goto LABEL_6;
LABEL_47:
          a1 = *(_QWORD *)(a1 + 24);
          goto LABEL_30;
        case 12LL:
          if ( *(_BYTE *)(a1 + 8) != 13 || *(_DWORD *)(a1 + 12) != (_DWORD)v8 )
            goto LABEL_6;
          v27 = 0;
          if ( !(_DWORD)v8 )
            goto LABEL_53;
          while ( !(unsigned __int8)sub_15E2EC0(*(_QWORD *)(*(_QWORD *)(a1 + 16) + v27), a2, a3) )
          {
            v27 += 8;
            if ( v27 == 8LL * (unsigned int)v8 )
            {
LABEL_53:
              LOBYTE(v9) = 0;
              return v9;
            }
          }
          goto LABEL_6;
        case 13LL:
          v26 = *((unsigned int *)a3 + 2);
          v9 = (unsigned int)v8 >> 3;
          if ( (unsigned int)v26 <= (unsigned int)v9 )
          {
            if ( (unsigned int)v26 >= *((_DWORD *)a3 + 3) )
            {
              sub_16CD150(a3, a3 + 2, 0, 8);
              v26 = *((unsigned int *)a3 + 2);
            }
            *(_QWORD *)(*a3 + 8 * v26) = a1;
            v28 = dword_4295390[v8 & 7];
            ++*((_DWORD *)a3 + 2);
            __asm { jmp     rax }
          }
          LOBYTE(v9) = *(_QWORD *)(*a3 + 8 * v9) != a1;
          return v9;
        case 14LL:
          v24 = (unsigned int)v8 >> 3;
          if ( *((_DWORD *)a3 + 2) <= (unsigned int)v24 )
            goto LABEL_6;
          v20 = *(_QWORD **)(*a3 + 8 * v24);
          v25 = *((_BYTE *)v20 + 8);
          if ( v25 == 16 )
          {
            v22 = 2 * (unsigned int)sub_1643030(v20[3]);
            goto LABEL_38;
          }
          if ( v25 != 11 )
            goto LABEL_6;
          v9 = sub_1644900(*v20, (unsigned int)(2 * (*((_DWORD *)v20 + 2) >> 8)));
          goto LABEL_39;
        case 15LL:
          v19 = (unsigned int)v8 >> 3;
          if ( *((_DWORD *)a3 + 2) <= (unsigned int)v19 )
            goto LABEL_6;
          v20 = *(_QWORD **)(*a3 + 8 * v19);
          v21 = *((_BYTE *)v20 + 8);
          if ( v21 == 16 )
          {
            v22 = (unsigned int)sub_1643030(v20[3]) >> 1;
LABEL_38:
            v23 = sub_1644900(*v20, v22);
            v9 = sub_16463B0(v23, v20[4]);
            goto LABEL_39;
          }
          if ( v21 == 11 )
          {
            v9 = sub_1644900(*v20, *((_DWORD *)v20 + 2) >> 9);
LABEL_39:
            LOBYTE(v9) = v9 != a1;
            return v9;
          }
LABEL_6:
          LOBYTE(v9) = 1;
          break;
        case 16LL:
          v17 = (unsigned int)v8 >> 3;
          if ( *((_DWORD *)a3 + 2) <= (unsigned int)v17 )
            goto LABEL_6;
          v18 = *(_QWORD *)(*a3 + 8 * v17);
          if ( *(_BYTE *)(v18 + 8) != 16 )
            goto LABEL_6;
          LOBYTE(v9) = a1 != sub_16463B0(*(_QWORD *)(v18 + 24), *(_DWORD *)(v18 + 32) >> 1);
          return v9;
        case 17LL:
          v15 = (unsigned int)v8 >> 3;
          if ( *((_DWORD *)a3 + 2) <= (unsigned int)v15 )
            goto LABEL_6;
          v16 = *(_QWORD *)(*a3 + 8 * v15);
          if ( *(_BYTE *)(v16 + 8) != 16 || *(_BYTE *)(a1 + 8) != 16 || *(_DWORD *)(a1 + 32) != *(_DWORD *)(v16 + 32) )
            goto LABEL_6;
          a1 = **(_QWORD **)(a1 + 16);
LABEL_30:
          if ( !v3 )
            goto LABEL_6;
          continue;
        case 18LL:
          v14 = (unsigned int)v8 >> 3;
          if ( *((_DWORD *)a3 + 2) <= (unsigned int)v14 || *(_BYTE *)(a1 + 8) != 15 )
            goto LABEL_6;
          LOBYTE(v9) = *(_QWORD *)(*a3 + 8 * v14) != *(_QWORD *)(a1 + 24);
          return v9;
        case 19LL:
          v13 = (unsigned int)v8 >> 3;
          if ( *((_DWORD *)a3 + 2) <= (unsigned int)v13 )
            goto LABEL_6;
          v9 = *(_QWORD *)(*a3 + 8 * v13);
          if ( *(_BYTE *)(v9 + 8) != 16 || *(_BYTE *)(a1 + 8) != 15 )
            goto LABEL_6;
          LOBYTE(v9) = *(_QWORD *)(v9 + 24) != *(_QWORD *)(a1 + 24);
          return v9;
        case 20LL:
          v10 = *((unsigned int *)a3 + 2);
          if ( (unsigned int)v10 <= (unsigned __int16)v8 )
            goto LABEL_6;
          if ( (unsigned int)v10 >= *((_DWORD *)a3 + 3) )
          {
            sub_16CD150(a3, a3 + 2, 0, 8);
            v10 = *((unsigned int *)a3 + 2);
          }
          *(_QWORD *)(*a3 + 8 * v10) = a1;
          v11 = *a3;
          ++*((_DWORD *)a3 + 2);
          v9 = *(_QWORD *)(v11 + 8LL * (unsigned __int16)v8);
          if ( *(_BYTE *)(v9 + 8) != 16 )
            goto LABEL_6;
          if ( *(_BYTE *)(a1 + 8) != 16 )
            goto LABEL_6;
          if ( *(_DWORD *)(a1 + 32) != *(_DWORD *)(v9 + 32) )
            goto LABEL_6;
          v12 = **(_QWORD **)(a1 + 16);
          if ( *(_BYTE *)(v12 + 8) != 15 )
            goto LABEL_6;
          LOBYTE(v9) = **(_QWORD **)(v9 + 16) != *(_QWORD *)(v12 + 24);
          return v9;
      }
      break;
    }
  }
  else
  {
    LOBYTE(v9) = 1;
  }
  return v9;
}
