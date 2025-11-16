// Function: sub_20D2E80
// Address: 0x20d2e80
//
_BOOL8 __fastcall sub_20D2E80(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v12; // rbx
  __int64 v13; // rax
  __int64 v14; // rdi
  __int64 v15; // rsi
  __int64 v16; // rax
  __int64 v17; // rbx
  __int64 v18; // r13
  __int64 v19; // rax
  __int64 v20; // rdi
  double v21; // xmm4_8
  double v22; // xmm5_8
  __int64 *v23; // rax
  unsigned __int64 v24; // rbx
  __int64 v25; // rsi
  __int64 v26; // rcx
  __int64 v28; // r13
  unsigned __int64 v29; // r15
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // r15
  __int64 v33; // rcx
  __int64 v34; // [rsp+8h] [rbp-48h]
  __int64 v35; // [rsp+10h] [rbp-40h]
  unsigned __int64 v36; // [rsp+18h] [rbp-38h]

  v12 = 1;
  v13 = sub_15F2050(a2);
  v14 = sub_1632FA0(v13);
  v15 = **(_QWORD **)(a2 - 48);
  while ( 2 )
  {
    switch ( *(_BYTE *)(v15 + 8) )
    {
      case 0:
      case 8:
      case 0xA:
      case 0xC:
      case 0x10:
        v30 = *(_QWORD *)(v15 + 32);
        v15 = *(_QWORD *)(v15 + 24);
        v12 *= v30;
        continue;
      case 1:
        v16 = 16;
        break;
      case 2:
        v16 = 32;
        break;
      case 3:
      case 9:
        v16 = 64;
        break;
      case 4:
        v16 = 80;
        break;
      case 5:
      case 6:
        v16 = 128;
        break;
      case 7:
        v16 = 8 * (unsigned int)sub_15A9520(v14, 0);
        break;
      case 0xB:
        v16 = *(_DWORD *)(v15 + 8) >> 8;
        break;
      case 0xD:
        v16 = 8LL * *(_QWORD *)sub_15A9930(v14, v15);
        break;
      case 0xE:
        v28 = *(_QWORD *)(v15 + 32);
        v35 = *(_QWORD *)(v15 + 24);
        v29 = (unsigned int)sub_15A9FE0(v14, v35);
        v16 = 8 * v29 * v28 * ((v29 + ((unsigned __int64)(sub_127FA20(v14, v35) + 7) >> 3) - 1) / v29);
        break;
      case 0xF:
        v16 = 8 * (unsigned int)sub_15A9520(v14, *(_DWORD *)(v15 + 8) >> 8);
        break;
    }
    break;
  }
  v17 = v16 * v12;
  v18 = 1;
  v19 = sub_15F2050(a2);
  v20 = sub_1632FA0(v19);
  v23 = *(__int64 **)(a2 - 48);
  v24 = (unsigned __int64)(v17 + 7) >> 3;
  v25 = *v23;
  while ( 1 )
  {
    switch ( *(_BYTE *)(v25 + 8) )
    {
      case 1:
        v26 = 16;
        return sub_20D1500(
                 a1,
                 a2,
                 v24,
                 (unsigned __int64)(v26 * v18 + 7) >> 3,
                 *(_QWORD *)(a2 - 72),
                 *(_QWORD **)(a2 - 24),
                 a3,
                 a4,
                 a5,
                 a6,
                 v21,
                 v22,
                 a9,
                 a10,
                 v23,
                 (*(unsigned __int16 *)(a2 + 18) >> 2) & 7,
                 (unsigned __int8)*(_WORD *)(a2 + 18) >> 5,
                 dword_430AB90);
      case 2:
        v26 = 32;
        return sub_20D1500(
                 a1,
                 a2,
                 v24,
                 (unsigned __int64)(v26 * v18 + 7) >> 3,
                 *(_QWORD *)(a2 - 72),
                 *(_QWORD **)(a2 - 24),
                 a3,
                 a4,
                 a5,
                 a6,
                 v21,
                 v22,
                 a9,
                 a10,
                 v23,
                 (*(unsigned __int16 *)(a2 + 18) >> 2) & 7,
                 (unsigned __int8)*(_WORD *)(a2 + 18) >> 5,
                 dword_430AB90);
      case 3:
      case 9:
        v26 = 64;
        return sub_20D1500(
                 a1,
                 a2,
                 v24,
                 (unsigned __int64)(v26 * v18 + 7) >> 3,
                 *(_QWORD *)(a2 - 72),
                 *(_QWORD **)(a2 - 24),
                 a3,
                 a4,
                 a5,
                 a6,
                 v21,
                 v22,
                 a9,
                 a10,
                 v23,
                 (*(unsigned __int16 *)(a2 + 18) >> 2) & 7,
                 (unsigned __int8)*(_WORD *)(a2 + 18) >> 5,
                 dword_430AB90);
      case 4:
        v26 = 80;
        return sub_20D1500(
                 a1,
                 a2,
                 v24,
                 (unsigned __int64)(v26 * v18 + 7) >> 3,
                 *(_QWORD *)(a2 - 72),
                 *(_QWORD **)(a2 - 24),
                 a3,
                 a4,
                 a5,
                 a6,
                 v21,
                 v22,
                 a9,
                 a10,
                 v23,
                 (*(unsigned __int16 *)(a2 + 18) >> 2) & 7,
                 (unsigned __int8)*(_WORD *)(a2 + 18) >> 5,
                 dword_430AB90);
      case 5:
      case 6:
        v26 = 128;
        return sub_20D1500(
                 a1,
                 a2,
                 v24,
                 (unsigned __int64)(v26 * v18 + 7) >> 3,
                 *(_QWORD *)(a2 - 72),
                 *(_QWORD **)(a2 - 24),
                 a3,
                 a4,
                 a5,
                 a6,
                 v21,
                 v22,
                 a9,
                 a10,
                 v23,
                 (*(unsigned __int16 *)(a2 + 18) >> 2) & 7,
                 (unsigned __int8)*(_WORD *)(a2 + 18) >> 5,
                 dword_430AB90);
      case 7:
        v26 = 8 * (unsigned int)sub_15A9520(v20, 0);
        v23 = *(__int64 **)(a2 - 48);
        return sub_20D1500(
                 a1,
                 a2,
                 v24,
                 (unsigned __int64)(v26 * v18 + 7) >> 3,
                 *(_QWORD *)(a2 - 72),
                 *(_QWORD **)(a2 - 24),
                 a3,
                 a4,
                 a5,
                 a6,
                 v21,
                 v22,
                 a9,
                 a10,
                 v23,
                 (*(unsigned __int16 *)(a2 + 18) >> 2) & 7,
                 (unsigned __int8)*(_WORD *)(a2 + 18) >> 5,
                 dword_430AB90);
      case 0xB:
        v26 = *(_DWORD *)(v25 + 8) >> 8;
        return sub_20D1500(
                 a1,
                 a2,
                 v24,
                 (unsigned __int64)(v26 * v18 + 7) >> 3,
                 *(_QWORD *)(a2 - 72),
                 *(_QWORD **)(a2 - 24),
                 a3,
                 a4,
                 a5,
                 a6,
                 v21,
                 v22,
                 a9,
                 a10,
                 v23,
                 (*(unsigned __int16 *)(a2 + 18) >> 2) & 7,
                 (unsigned __int8)*(_WORD *)(a2 + 18) >> 5,
                 dword_430AB90);
      case 0xD:
        v33 = *(_QWORD *)sub_15A9930(v20, v25);
        v23 = *(__int64 **)(a2 - 48);
        v26 = 8 * v33;
        return sub_20D1500(
                 a1,
                 a2,
                 v24,
                 (unsigned __int64)(v26 * v18 + 7) >> 3,
                 *(_QWORD *)(a2 - 72),
                 *(_QWORD **)(a2 - 24),
                 a3,
                 a4,
                 a5,
                 a6,
                 v21,
                 v22,
                 a9,
                 a10,
                 v23,
                 (*(unsigned __int16 *)(a2 + 18) >> 2) & 7,
                 (unsigned __int8)*(_WORD *)(a2 + 18) >> 5,
                 dword_430AB90);
      case 0xE:
        v32 = *(_QWORD *)(v25 + 32);
        v34 = *(_QWORD *)(v25 + 24);
        v36 = (unsigned int)sub_15A9FE0(v20, v34);
        v26 = 8 * v32 * v36 * ((v36 + ((unsigned __int64)(sub_127FA20(v20, v34) + 7) >> 3) - 1) / v36);
        v23 = *(__int64 **)(a2 - 48);
        return sub_20D1500(
                 a1,
                 a2,
                 v24,
                 (unsigned __int64)(v26 * v18 + 7) >> 3,
                 *(_QWORD *)(a2 - 72),
                 *(_QWORD **)(a2 - 24),
                 a3,
                 a4,
                 a5,
                 a6,
                 v21,
                 v22,
                 a9,
                 a10,
                 v23,
                 (*(unsigned __int16 *)(a2 + 18) >> 2) & 7,
                 (unsigned __int8)*(_WORD *)(a2 + 18) >> 5,
                 dword_430AB90);
      case 0xF:
        v26 = 8 * (unsigned int)sub_15A9520(v20, *(_DWORD *)(v25 + 8) >> 8);
        v23 = *(__int64 **)(a2 - 48);
        return sub_20D1500(
                 a1,
                 a2,
                 v24,
                 (unsigned __int64)(v26 * v18 + 7) >> 3,
                 *(_QWORD *)(a2 - 72),
                 *(_QWORD **)(a2 - 24),
                 a3,
                 a4,
                 a5,
                 a6,
                 v21,
                 v22,
                 a9,
                 a10,
                 v23,
                 (*(unsigned __int16 *)(a2 + 18) >> 2) & 7,
                 (unsigned __int8)*(_WORD *)(a2 + 18) >> 5,
                 dword_430AB90);
      case 0x10:
        v31 = *(_QWORD *)(v25 + 32);
        v25 = *(_QWORD *)(v25 + 24);
        v18 *= v31;
        break;
      default:
        BUG();
    }
  }
}
