// Function: sub_7D9920
// Address: 0x7d9920
//
void __fastcall sub_7D9920(_QWORD *a1, const __m128i *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __m128i *v6; // r13
  __int64 j; // r12
  __int64 v8; // r14
  __int64 v9; // rbx
  __int64 v10; // rax
  __int64 v11; // r13
  char v12; // dl
  char v13; // al
  unsigned int v14; // eax
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // rbx
  unsigned int i; // r14d
  __int64 v21; // rax
  __int64 v22; // rax
  _BYTE *v23; // rax
  int v24; // [rsp+Ch] [rbp-A4h] BYREF
  _BYTE v25[32]; // [rsp+10h] [rbp-A0h] BYREF
  _DWORD v26[32]; // [rsp+30h] [rbp-80h] BYREF

  v6 = (__m128i *)a2;
  while ( 2 )
  {
    *((_BYTE *)a1 - 8) |= 8u;
    switch ( *((_BYTE *)a1 + 24) )
    {
      case 1:
        v13 = *((_BYTE *)a1 + 56);
        if ( v13 == 103 )
        {
          sub_7F2B50(a1, 0);
        }
        else if ( (unsigned __int8)(v13 - 87) <= 1u )
        {
          sub_7F36F0(a1);
        }
        else
        {
          v14 = sub_7E6AE0(a1);
          v19 = a1[9];
          for ( i = v14; v19; i >>= 1 )
          {
            if ( (i & 1) != 0 )
            {
              a2 = 0;
              sub_7D98E0(v19, 0);
            }
            else
            {
              sub_7D9DD0(v19);
            }
            v19 = *(_QWORD *)(v19 + 16);
          }
          sub_7D9310(a1, (__int64)a2, v15, v16, v17, v18);
          if ( *((_BYTE *)a1 + 24) == 1 && (unsigned __int8)(*((_BYTE *)a1 + 56) - 105) <= 4u )
          {
            sub_825720(a1);
            if ( *((_BYTE *)a1 + 24) == 1 && (unsigned __int8)(*((_BYTE *)a1 + 56) - 105) <= 4u )
            {
              if ( unk_4D04380 )
                sub_76EF80(a1, v6, v26);
            }
          }
        }
        return;
      case 2:
        sub_7D8EC0(a1);
        return;
      case 3:
        if ( (*((_BYTE *)a1 + 25) & 1) != 0 )
        {
          v21 = a1[7];
          if ( *(char *)(v21 + 169) < 0 )
            *(_BYTE *)(v21 + 171) |= 0x20u;
        }
        return;
      case 4:
      case 0x10:
      case 0x15:
      case 0x18:
        return;
      case 5:
        v8 = a1[7];
        v9 = *a1;
        if ( dword_4D047EC && (unsigned int)sub_8DD010(*a1) )
        {
          if ( *(_BYTE *)(v9 + 140) != 12 || !*(_QWORD *)(v9 + 8) )
            sub_8DD360(v9);
          v22 = sub_7E7CA0(v9);
          *(_QWORD *)(v8 + 8) = v22;
          v11 = v22;
          if ( dword_4F077C4 == 2 || unk_4F07778 <= 201111 || *(_BYTE *)(v9 + 140) != 12 || *(_BYTE *)(v9 + 184) != 8 )
          {
LABEL_13:
            *(_BYTE *)(v11 + 173) |= 2u;
            goto LABEL_14;
          }
          v12 = 1;
LABEL_12:
          *(_QWORD *)(v11 + 104) = *(_QWORD *)(v9 + 104);
          *(_QWORD *)(v9 + 104) = 0;
          if ( !v12 )
            goto LABEL_14;
          goto LABEL_13;
        }
        v10 = sub_7E7CA0(v9);
        *(_QWORD *)(v8 + 8) = v10;
        v11 = v10;
        if ( dword_4F077C4 != 2 && unk_4F07778 > 201111 && *(_BYTE *)(v9 + 140) == 12 && *(_BYTE *)(v9 + 184) == 8 )
        {
          v12 = 0;
          goto LABEL_12;
        }
LABEL_14:
        sub_7264E0((__int64)a1, 3);
        a1[7] = v11;
        sub_7E1780(a1, v25);
        sub_7F9080(v11, v26);
        sub_7FEC50(v8, (unsigned int)v26, 0, 0, 0, 0, (__int64)v25, (__int64)&v24, 0);
        if ( (*(_BYTE *)(v11 + 173) & 2) != 0 )
        {
          v23 = sub_726B30(22);
          v23[72] = 0;
          *((_QWORD *)v23 + 10) = v11;
          sub_7FCA00(v23);
        }
        if ( v24 )
          sub_7FCA60(v11, v8);
        if ( *(_BYTE *)(v11 + 177) == 3 && *(_BYTE *)(v11 + 136) > 2u )
          sub_7FBBC0(v11, a1);
        return;
      case 0xC:
        sub_7D9DE0(a1);
        return;
      case 0x11:
        sub_7EE3B0(a1);
        return;
      case 0x12:
        sub_7E6B10(a1);
        return;
      case 0x14:
        if ( (*(_BYTE *)(a1[7] + 201LL) & 1) != 0 )
          sub_7E52E0(a1);
        return;
      case 0x17:
        sub_7F3860(a1);
        return;
      case 0x1A:
        a2 = (const __m128i *)a1[8];
        if ( (a2[1].m128i_i8[9] & 1) != 0 && (*((_BYTE *)a1 + 25) & 1) == 0 )
          a2 = (const __m128i *)sub_731370(a1[8], (__int64)a2, a3, a4, a5, a6);
        a2[1].m128i_i64[0] = a1[2];
        sub_730620((__int64)a1, a2);
        continue;
      case 0x1B:
        for ( j = a1[7]; j; j = *(_QWORD *)(j + 16) )
          sub_7D9DD0(j);
        return;
      default:
        sub_721090();
    }
  }
}
