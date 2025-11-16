// Function: sub_141F110
// Address: 0x141f110
//
__m128i *__fastcall sub_141F110(__m128i *a1, __int64 a2)
{
  __int64 v3; // rbx
  __int64 v4; // rax
  __int64 v5; // r15
  __int64 v6; // rsi
  __int64 v7; // rax
  __int64 v8; // rbx
  __int64 v9; // rdx
  __m128i v10; // xmm0
  __int64 v12; // r14
  unsigned int v13; // eax
  __int64 v14; // rsi
  __int64 v15; // rcx
  unsigned __int64 v16; // r8
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rsi
  int v20; // eax
  __int64 v21; // rax
  _QWORD *v22; // rax
  __int64 v23; // [rsp+18h] [rbp-68h]
  __int64 v24; // [rsp+20h] [rbp-60h]
  __int64 v25; // [rsp+20h] [rbp-60h]
  __int64 v26; // [rsp+28h] [rbp-58h]
  unsigned __int64 v27; // [rsp+28h] [rbp-58h]
  unsigned __int64 v28; // [rsp+28h] [rbp-58h]
  __m128i v29; // [rsp+30h] [rbp-50h] BYREF
  __int64 v30; // [rsp+40h] [rbp-40h]

  v3 = 1;
  v29 = 0u;
  v30 = 0;
  sub_14A8180(a2, &v29, 0);
  v4 = sub_15F2050(a2);
  v5 = sub_1632FA0(v4);
  v6 = **(_QWORD **)(a2 - 48);
  while ( 2 )
  {
    switch ( *(_BYTE *)(v6 + 8) )
    {
      case 0:
      case 8:
      case 0xA:
      case 0xC:
      case 0x10:
        v17 = *(_QWORD *)(v6 + 32);
        v6 = *(_QWORD *)(v6 + 24);
        v3 *= v17;
        continue;
      case 1:
        v7 = 16;
        break;
      case 2:
        v7 = 32;
        break;
      case 3:
      case 9:
        v7 = 64;
        break;
      case 4:
        v7 = 80;
        break;
      case 5:
      case 6:
        v7 = 128;
        break;
      case 7:
        v7 = 8 * (unsigned int)sub_15A9520(v5, 0);
        break;
      case 0xB:
        v7 = *(_DWORD *)(v6 + 8) >> 8;
        break;
      case 0xD:
        v7 = 8LL * *(_QWORD *)sub_15A9930(v5, v6);
        break;
      case 0xE:
        v12 = *(_QWORD *)(v6 + 32);
        v26 = *(_QWORD *)(v6 + 24);
        v13 = sub_15A9FE0(v5, v26);
        v14 = v26;
        v15 = 1;
        v16 = v13;
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
              v15 *= v21;
              continue;
            case 1:
              v18 = 16;
              goto LABEL_17;
            case 2:
              v18 = 32;
              goto LABEL_17;
            case 3:
            case 9:
              v18 = 64;
              goto LABEL_17;
            case 4:
              v18 = 80;
              goto LABEL_17;
            case 5:
            case 6:
              v18 = 128;
              goto LABEL_17;
            case 7:
              v24 = v15;
              v19 = 0;
              v27 = v16;
              goto LABEL_23;
            case 0xB:
              v18 = *(_DWORD *)(v14 + 8) >> 8;
              goto LABEL_17;
            case 0xD:
              v25 = v15;
              v28 = v16;
              v22 = (_QWORD *)sub_15A9930(v5, v14);
              v16 = v28;
              v15 = v25;
              v18 = 8LL * *v22;
              goto LABEL_17;
            case 0xE:
              v23 = *(_QWORD *)(v14 + 24);
              sub_15A9FE0(v5, v23);
              sub_127FA20(v5, v23);
              JUMPOUT(0x141F36E);
            case 0xF:
              v24 = v15;
              v27 = v16;
              v19 = *(_DWORD *)(v14 + 8) >> 8;
LABEL_23:
              v20 = sub_15A9520(v5, v19);
              v16 = v27;
              v15 = v24;
              v18 = (unsigned int)(8 * v20);
LABEL_17:
              v7 = 8 * v16 * v12 * ((v16 + ((unsigned __int64)(v18 * v15 + 7) >> 3) - 1) / v16);
              break;
          }
          break;
        }
        break;
      case 0xF:
        v7 = 8 * (unsigned int)sub_15A9520(v5, *(_DWORD *)(v6 + 8) >> 8);
        break;
    }
    break;
  }
  v8 = v7 * v3;
  v9 = *(_QWORD *)(a2 - 72);
  v10 = _mm_loadu_si128(&v29);
  a1[2].m128i_i64[0] = v30;
  a1->m128i_i64[0] = v9;
  a1[1] = v10;
  a1->m128i_i64[1] = (unsigned __int64)(v8 + 7) >> 3;
  return a1;
}
