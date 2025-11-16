// Function: sub_135C800
// Address: 0x135c800
//
__int64 __fastcall sub_135C800(__int64 a1, __int64 a2)
{
  __int64 v4; // rbx
  __int64 v5; // rax
  __int64 v6; // rdi
  __int64 v7; // rsi
  __int64 result; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // r14
  unsigned int v12; // eax
  __int64 v13; // r9
  unsigned __int64 v14; // r8
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // [rsp+28h] [rbp-58h]
  __m128i v18; // [rsp+30h] [rbp-50h] BYREF
  __int64 v19; // [rsp+40h] [rbp-40h]

  if ( byte_42880A0[8 * ((*(unsigned __int16 *)(a2 + 18) >> 7) & 7) + 2] )
    return sub_135AA40(a1, a2);
  v4 = 1;
  v18 = 0u;
  v19 = 0;
  sub_14A8180(a2, &v18, 0);
  v5 = sub_15F2050(a2);
  v6 = sub_1632FA0(v5);
  v7 = **(_QWORD **)(a2 - 48);
  while ( 2 )
  {
    switch ( *(_BYTE *)(v7 + 8) )
    {
      case 0:
      case 8:
      case 0xA:
      case 0xC:
      case 0x10:
        v10 = *(_QWORD *)(v7 + 32);
        v7 = *(_QWORD *)(v7 + 24);
        v4 *= v10;
        continue;
      case 1:
        v9 = 16;
        break;
      case 2:
        v9 = 32;
        break;
      case 3:
      case 9:
        v9 = 64;
        break;
      case 4:
        v9 = 80;
        break;
      case 5:
      case 6:
        v9 = 128;
        break;
      case 7:
        v9 = 8 * (unsigned int)sub_15A9520(v6, 0);
        break;
      case 0xB:
        v9 = *(_DWORD *)(v7 + 8) >> 8;
        break;
      case 0xD:
        v9 = 8LL * *(_QWORD *)sub_15A9930(v6, v7);
        break;
      case 0xE:
        v11 = *(_QWORD *)(v7 + 24);
        v17 = *(_QWORD *)(v7 + 32);
        v12 = sub_15A9FE0(v6, v11);
        v13 = 1;
        v14 = v12;
        while ( 2 )
        {
          switch ( *(_BYTE *)(v11 + 8) )
          {
            case 0:
              v16 = *(_QWORD *)(v11 + 32);
              v11 = *(_QWORD *)(v11 + 24);
              v13 *= v16;
              continue;
            case 1:
              v15 = 16;
              break;
            case 2:
              v15 = 32;
              break;
            case 3:
              v15 = 64;
              break;
            case 4:
              v15 = 80;
              break;
            case 5:
            case 6:
              v15 = 128;
              break;
          }
          break;
        }
        v9 = 8 * v14 * v17 * ((v14 + ((unsigned __int64)(v15 * v13 + 7) >> 3) - 1) / v14);
        break;
      case 0xF:
        v9 = 8 * (unsigned int)sub_15A9520(v6, *(_DWORD *)(v7 + 8) >> 8);
        break;
    }
    break;
  }
  result = sub_135C460(a1, *(_QWORD *)(a2 - 24), (unsigned __int64)(v9 * v4 + 7) >> 3, &v18, 2);
  if ( (*(_BYTE *)(a2 + 18) & 1) != 0 )
    *(_BYTE *)(result + 67) |= 0x80u;
  return result;
}
