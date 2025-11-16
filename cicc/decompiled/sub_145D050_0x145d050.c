// Function: sub_145D050
// Address: 0x145d050
//
__int64 __fastcall sub_145D050(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rbx
  __int64 v6; // rdi
  unsigned __int64 v7; // r14
  __int64 v8; // rax
  unsigned __int64 v10; // r15
  __int64 v11; // rax
  __int64 v12; // [rsp+8h] [rbp-48h]
  __int64 v13; // [rsp+10h] [rbp-40h]
  __int64 v14; // [rsp+18h] [rbp-38h]
  __int64 v15; // [rsp+18h] [rbp-38h]

  v5 = 1;
  v14 = sub_1632FA0(*(_QWORD *)(*(_QWORD *)(a1 + 24) + 40LL));
  v6 = v14;
  v7 = (unsigned int)sub_15A9FE0(v14, a3);
  while ( 2 )
  {
    switch ( *(_BYTE *)(a3 + 8) )
    {
      case 0:
      case 8:
      case 0xA:
      case 0xC:
      case 0x10:
        v11 = *(_QWORD *)(a3 + 32);
        a3 = *(_QWORD *)(a3 + 24);
        v5 *= v11;
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
        v8 = 8 * (unsigned int)sub_15A9520(v14, 0);
        break;
      case 0xB:
        v8 = *(_DWORD *)(a3 + 8) >> 8;
        break;
      case 0xD:
        v8 = 8LL * *(_QWORD *)sub_15A9930(v14, a3);
        break;
      case 0xE:
        v13 = v14;
        v12 = *(_QWORD *)(a3 + 24);
        v15 = *(_QWORD *)(a3 + 32);
        v10 = (unsigned int)sub_15A9FE0(v6, v12);
        v8 = 8 * v15 * v10 * ((v10 + ((unsigned __int64)(sub_127FA20(v13, v12) + 7) >> 3) - 1) / v10);
        break;
      case 0xF:
        v8 = 8 * (unsigned int)sub_15A9520(v14, *(_DWORD *)(a3 + 8) >> 8);
        break;
    }
    break;
  }
  return sub_145CF80(a1, a2, v7 * ((v7 + ((unsigned __int64)(v8 * v5 + 7) >> 3) - 1) / v7), 0);
}
