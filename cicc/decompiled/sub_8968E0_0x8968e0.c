// Function: sub_8968E0
// Address: 0x8968e0
//
void __fastcall sub_8968E0(__int64 a1, char a2)
{
  __int64 v2; // r14
  __int64 v4; // r15
  _QWORD *v5; // rax
  __int64 v6; // rbx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  char v10; // dl
  int v11; // esi
  char v12; // al
  char v13; // dl
  __int64 v14; // rdx
  __int64 i; // rax
  __int64 v16; // r13
  _QWORD *v17; // rax
  __int64 v18; // rax
  __int64 v19; // rcx
  char v20; // r15
  __int64 v21; // rax
  __int64 v22; // rbx
  char v23; // r15
  __int64 v24; // rax
  char v25; // dl
  __int64 v26; // [rsp-50h] [rbp-50h]
  __int64 v27; // [rsp-48h] [rbp-48h]
  __int64 v28; // [rsp-40h] [rbp-40h]
  __int64 v29; // [rsp-40h] [rbp-40h]

  if ( (*(_BYTE *)(a1 + 81) & 0x10) != 0 )
  {
    v2 = *(_QWORD *)(a1 + 88);
    if ( (*(_BYTE *)(v2 + 89) & 4) != 0 )
    {
      v4 = *(_QWORD *)(*(_QWORD *)(v2 + 40) + 32LL);
      if ( dword_4F04C44 == -1 && (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) == 0 )
      {
        v21 = sub_6011C0(a1);
        v22 = v21;
        if ( v21 )
        {
          switch ( *(_BYTE *)(v21 + 80) )
          {
            case 4:
            case 5:
              v16 = *(_QWORD *)(*(_QWORD *)(v21 + 96) + 80LL);
              break;
            case 6:
              v16 = *(_QWORD *)(*(_QWORD *)(v21 + 96) + 32LL);
              break;
            case 9:
            case 0xA:
              v16 = *(_QWORD *)(*(_QWORD *)(v21 + 96) + 56LL);
              break;
            case 0x13:
            case 0x14:
            case 0x15:
            case 0x16:
              v16 = *(_QWORD *)(v21 + 88);
              break;
            default:
              sub_878440()[1] = a1;
              BUG();
          }
          v28 = *(_QWORD *)(a1 + 96);
          v17 = sub_878440();
          v17[1] = a1;
          *v17 = *(_QWORD *)(v16 + 168);
          *(_QWORD *)(v16 + 168) = v17;
          *(_QWORD *)(v28 + 104) = v22;
          LOBYTE(v17) = *(_BYTE *)(v2 + 177) | 0x10;
          *(_BYTE *)(v2 + 177) = (_BYTE)v17;
          LOBYTE(v17) = *(_BYTE *)(v4 + 177) & 0x20 | (unsigned __int8)v17 & 0xDF;
          *(_BYTE *)(v2 + 177) = (_BYTE)v17;
          *(_BYTE *)(v2 + 177) = *(_BYTE *)(v4 + 177) & 0x40 | (unsigned __int8)v17 & 0xBF;
          sub_890050(v16, v2);
          *(_QWORD *)(*(_QWORD *)(v2 + 168) + 160LL) = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v22 + 88) + 168LL) + 160LL);
        }
      }
      else
      {
        v5 = sub_727340();
        *((_BYTE *)v5 + 120) = 6;
        v6 = (__int64)v5;
        sub_877D80((__int64)v5, (__int64 *)a1);
        sub_877E20(0, v6, v4, v7, v8, v9);
        v10 = 0;
        if ( *(char *)(v4 + 177) < 0 )
          v10 = *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v4 + 168) + 160LL) + 121LL) & 1;
        v11 = dword_4F04C64;
        v12 = v10 | *(_BYTE *)(v6 + 121) & 0xFE;
        v13 = *(_BYTE *)(v6 + 88);
        *(_BYTE *)(v6 + 121) = v12;
        *(_BYTE *)(v6 + 88) = v13 & 0x8C | *(_BYTE *)(v2 + 88) & 3 | 0x20;
        sub_7344C0(v6, v11);
        *(_QWORD *)(v6 + 200) = v6;
        if ( word_4F06418[0] == 55 || word_4F06418[0] == 73 )
          *(_QWORD *)(v6 + 208) = v6;
        *(_QWORD *)(*(_QWORD *)(v2 + 168) + 160LL) = v6;
        v14 = *(_QWORD *)(a1 + 64);
        for ( i = v14; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
          ;
        if ( *(char *)(v14 + 177) < 0 )
        {
          v19 = *(_QWORD *)(a1 + 88);
          if ( (*(_BYTE *)(v14 + 178) & 4) != 0 )
          {
            *(_BYTE *)(v19 + 178) |= 1u;
            v20 = 1;
          }
          else
          {
            v26 = *(_QWORD *)(a1 + 88);
            v29 = *(_QWORD *)(a1 + 96);
            v23 = *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)i + 96LL) + 80LL) + 160LL) >> 7;
            v27 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)i + 96LL) + 80LL);
            *(_DWORD *)(v29 + 112) = dword_4F06650[0];
            v20 = v23 ^ 1;
            v24 = sub_87E420(*(_BYTE *)(a1 + 80));
            v19 = v26;
            v25 = *(_BYTE *)(v27 + 265);
            *(_BYTE *)(v24 + 264) = a2;
            *(_BYTE *)(v24 + 265) = v25 & 0x1C | *(_BYTE *)(v24 + 265) & 0xE3;
            *(_QWORD *)(v29 + 80) = v24;
            *(_QWORD *)(v29 + 104) = a1;
          }
          *(_BYTE *)(v19 + 177) = (32 * v20) | (v20 << 7) | *(_BYTE *)(v19 + 177) & 0x5F | 0x10;
        }
        switch ( *(_BYTE *)(a1 + 80) )
        {
          case 4:
          case 5:
            v18 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 80LL);
            goto LABEL_18;
          case 6:
            v18 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 32LL);
            goto LABEL_18;
          case 9:
          case 0xA:
            v18 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 56LL);
            goto LABEL_18;
          case 0x13:
          case 0x14:
          case 0x15:
          case 0x16:
            v18 = *(_QWORD *)(a1 + 88);
LABEL_18:
            if ( !dword_4F07590 )
            {
              if ( !v18 )
                goto LABEL_23;
              if ( *(char *)(v18 + 160) < 0 )
                *(_QWORD *)(v6 + 192) = v2;
              goto LABEL_22;
            }
            *(_QWORD *)(v6 + 192) = v2;
            if ( v18 )
            {
LABEL_22:
              *(_QWORD *)(v18 + 104) = v6;
              *(_QWORD *)(v18 + 176) = a1;
            }
LABEL_23:
            sub_878FA0((_QWORD *)a1);
            break;
          default:
            if ( dword_4F07590 )
              *(_QWORD *)(v6 + 192) = v2;
            goto LABEL_23;
        }
      }
    }
  }
}
