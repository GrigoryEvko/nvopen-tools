// Function: sub_1456C90
// Address: 0x1456c90
//
__int64 __fastcall sub_1456C90(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // rdi
  __int64 v4; // rbx
  __int64 v5; // rdi
  __int64 v6; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // r14
  __int64 v11; // r13
  unsigned __int64 v12; // r12

  v2 = a2;
  v3 = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 40LL);
  if ( *(_BYTE *)(a2 + 8) == 15 )
  {
    v6 = sub_1632FA0(v3);
    return (unsigned int)sub_15A95F0(v6, a2);
  }
  else
  {
    v4 = 1;
    v5 = sub_1632FA0(v3);
    while ( 2 )
    {
      switch ( *(_BYTE *)(v2 + 8) )
      {
        case 0:
        case 8:
        case 0xA:
        case 0xC:
        case 0x10:
          v9 = *(_QWORD *)(v2 + 32);
          v2 = *(_QWORD *)(v2 + 24);
          v4 *= v9;
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
          v8 = 8 * (unsigned int)sub_15A9520(v5, 0);
          break;
        case 0xB:
          v8 = *(_DWORD *)(v2 + 8) >> 8;
          break;
        case 0xD:
          v8 = 8LL * *(_QWORD *)sub_15A9930(v5, v2);
          break;
        case 0xE:
          v10 = *(_QWORD *)(v2 + 24);
          v11 = *(_QWORD *)(v2 + 32);
          v12 = (unsigned int)sub_15A9FE0(v5, v10);
          v8 = 8 * v11 * v12 * ((v12 + ((unsigned __int64)(sub_127FA20(v5, v10) + 7) >> 3) - 1) / v12);
          break;
        case 0xF:
          v8 = 8 * (unsigned int)sub_15A9520(v5, *(_DWORD *)(v2 + 8) >> 8);
          break;
      }
      break;
    }
    return v4 * v8;
  }
}
