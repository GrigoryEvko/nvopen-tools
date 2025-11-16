// Function: sub_135C4E0
// Address: 0x135c4e0
//
__int64 __fastcall sub_135C4E0(__int64 a1, __int64 a2)
{
  __int64 v4; // rbx
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rsi
  __int64 v8; // rdi
  __int64 result; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // r14
  unsigned int v13; // eax
  __int64 v14; // r9
  unsigned __int64 v15; // r8
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // [rsp+28h] [rbp-58h]
  __m128i v19; // [rsp+30h] [rbp-50h] BYREF
  __int64 v20; // [rsp+40h] [rbp-40h]

  if ( byte_42880A0[8 * ((*(unsigned __int16 *)(a2 + 18) >> 7) & 7) + 2] )
    return sub_135AA40(a1, a2);
  v4 = 1;
  v19 = 0u;
  v20 = 0;
  sub_14A8180(a2, &v19, 0);
  v5 = sub_15F2050(a2);
  v6 = sub_1632FA0(v5);
  v7 = *(_QWORD *)a2;
  v8 = v6;
  while ( 2 )
  {
    switch ( *(_BYTE *)(v7 + 8) )
    {
      case 0:
      case 8:
      case 0xA:
      case 0xC:
      case 0x10:
        v11 = *(_QWORD *)(v7 + 32);
        v7 = *(_QWORD *)(v7 + 24);
        v4 *= v11;
        continue;
      case 1:
        v10 = 16;
        break;
      case 2:
        v10 = 32;
        break;
      case 3:
      case 9:
        v10 = 64;
        break;
      case 4:
        v10 = 80;
        break;
      case 5:
      case 6:
        v10 = 128;
        break;
      case 7:
        v10 = 8 * (unsigned int)sub_15A9520(v8, 0);
        break;
      case 0xB:
        v10 = *(_DWORD *)(v7 + 8) >> 8;
        break;
      case 0xD:
        v10 = 8LL * *(_QWORD *)sub_15A9930(v8, v7);
        break;
      case 0xE:
        v12 = *(_QWORD *)(v7 + 24);
        v18 = *(_QWORD *)(v7 + 32);
        v13 = sub_15A9FE0(v8, v12);
        v14 = 1;
        v15 = v13;
        while ( 2 )
        {
          switch ( *(_BYTE *)(v12 + 8) )
          {
            case 0:
              v17 = *(_QWORD *)(v12 + 32);
              v12 = *(_QWORD *)(v12 + 24);
              v14 *= v17;
              continue;
            case 1:
              v16 = 16;
              break;
            case 2:
              v16 = 32;
              break;
            case 3:
              v16 = 64;
              break;
            case 4:
              v16 = 80;
              break;
            case 5:
            case 6:
              v16 = 128;
              break;
          }
          break;
        }
        v10 = 8 * v15 * v18 * ((v15 + ((unsigned __int64)(v16 * v14 + 7) >> 3) - 1) / v15);
        break;
      case 0xF:
        v10 = 8 * (unsigned int)sub_15A9520(v8, *(_DWORD *)(v7 + 8) >> 8);
        break;
    }
    break;
  }
  result = sub_135C460(a1, *(_QWORD *)(a2 - 24), (unsigned __int64)(v10 * v4 + 7) >> 3, &v19, 1);
  if ( (*(_BYTE *)(a2 + 18) & 1) != 0 )
    *(_BYTE *)(result + 67) |= 0x80u;
  return result;
}
