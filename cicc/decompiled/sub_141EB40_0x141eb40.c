// Function: sub_141EB40
// Address: 0x141eb40
//
__m128i *__fastcall sub_141EB40(__m128i *a1, __int64 *a2)
{
  __int64 v3; // rbx
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rsi
  __int64 v7; // r15
  __int64 v8; // rax
  __int64 v9; // rbx
  __int64 v10; // rdx
  __m128i v11; // xmm0
  __int64 v13; // r14
  unsigned int v14; // eax
  __int64 v15; // rsi
  __int64 v16; // rcx
  unsigned __int64 v17; // r8
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rsi
  int v21; // eax
  __int64 v22; // rax
  _QWORD *v23; // rax
  __int64 v24; // [rsp+18h] [rbp-68h]
  __int64 v25; // [rsp+20h] [rbp-60h]
  __int64 v26; // [rsp+20h] [rbp-60h]
  __int64 v27; // [rsp+28h] [rbp-58h]
  unsigned __int64 v28; // [rsp+28h] [rbp-58h]
  unsigned __int64 v29; // [rsp+28h] [rbp-58h]
  __m128i v30; // [rsp+30h] [rbp-50h] BYREF
  __int64 v31; // [rsp+40h] [rbp-40h]

  v3 = 1;
  v30 = 0u;
  v31 = 0;
  sub_14A8180(a2, &v30, 0);
  v4 = sub_15F2050(a2);
  v5 = sub_1632FA0(v4);
  v6 = *a2;
  v7 = v5;
  while ( 2 )
  {
    switch ( *(_BYTE *)(v6 + 8) )
    {
      case 0:
      case 8:
      case 0xA:
      case 0xC:
      case 0x10:
        v18 = *(_QWORD *)(v6 + 32);
        v6 = *(_QWORD *)(v6 + 24);
        v3 *= v18;
        continue;
      case 1:
        v8 = 16;
        break;
      case 2:
        v8 = 32;
        break;
      case 3:
      case 9:
        v8 = 64;
        break;
      case 4:
        v8 = 80;
        break;
      case 5:
      case 6:
        v8 = 128;
        break;
      case 7:
        v8 = 8 * (unsigned int)sub_15A9520(v7, 0);
        break;
      case 0xB:
        v8 = *(_DWORD *)(v6 + 8) >> 8;
        break;
      case 0xD:
        v8 = 8LL * *(_QWORD *)sub_15A9930(v7, v6);
        break;
      case 0xE:
        v13 = *(_QWORD *)(v6 + 32);
        v27 = *(_QWORD *)(v6 + 24);
        v14 = sub_15A9FE0(v7, v27);
        v15 = v27;
        v16 = 1;
        v17 = v14;
        while ( 2 )
        {
          switch ( *(_BYTE *)(v15 + 8) )
          {
            case 0:
            case 8:
            case 0xA:
            case 0xC:
            case 0x10:
              v22 = *(_QWORD *)(v15 + 32);
              v15 = *(_QWORD *)(v15 + 24);
              v16 *= v22;
              continue;
            case 1:
              v19 = 16;
              goto LABEL_17;
            case 2:
              v19 = 32;
              goto LABEL_17;
            case 3:
            case 9:
              v19 = 64;
              goto LABEL_17;
            case 4:
              v19 = 80;
              goto LABEL_17;
            case 5:
            case 6:
              v19 = 128;
              goto LABEL_17;
            case 7:
              v25 = v16;
              v20 = 0;
              v28 = v17;
              goto LABEL_23;
            case 0xB:
              v19 = *(_DWORD *)(v15 + 8) >> 8;
              goto LABEL_17;
            case 0xD:
              v26 = v16;
              v29 = v17;
              v23 = (_QWORD *)sub_15A9930(v7, v15);
              v17 = v29;
              v16 = v26;
              v19 = 8LL * *v23;
              goto LABEL_17;
            case 0xE:
              v24 = *(_QWORD *)(v15 + 24);
              sub_15A9FE0(v7, v24);
              sub_127FA20(v7, v24);
              JUMPOUT(0x141ED9E);
            case 0xF:
              v25 = v16;
              v28 = v17;
              v20 = *(_DWORD *)(v15 + 8) >> 8;
LABEL_23:
              v21 = sub_15A9520(v7, v20);
              v17 = v28;
              v16 = v25;
              v19 = (unsigned int)(8 * v21);
LABEL_17:
              v8 = 8 * v17 * v13 * ((v17 + ((unsigned __int64)(v19 * v16 + 7) >> 3) - 1) / v17);
              break;
          }
          break;
        }
        break;
      case 0xF:
        v8 = 8 * (unsigned int)sub_15A9520(v7, *(_DWORD *)(v6 + 8) >> 8);
        break;
    }
    break;
  }
  v9 = v8 * v3;
  v10 = *(a2 - 3);
  v11 = _mm_loadu_si128(&v30);
  a1[2].m128i_i64[0] = v31;
  a1->m128i_i64[0] = v10;
  a1[1] = v11;
  a1->m128i_i64[1] = (unsigned __int64)(v9 + 7) >> 3;
  return a1;
}
