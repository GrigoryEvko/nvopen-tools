// Function: sub_5ED6F0
// Address: 0x5ed6f0
//
__int64 __fastcall sub_5ED6F0(__int64 a1)
{
  __int64 v1; // r12
  __int64 v2; // rbx
  __int64 v3; // r13
  __int64 v4; // rdx
  __int64 v5; // rcx
  int v6; // eax
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 result; // rax
  __int64 v12; // [rsp+0h] [rbp-60h] BYREF
  __int64 v13; // [rsp+8h] [rbp-58h]
  __int64 v14; // [rsp+10h] [rbp-50h]
  __int64 v15; // [rsp+18h] [rbp-48h]
  __int64 v16; // [rsp+20h] [rbp-40h]
  __int64 v17; // [rsp+28h] [rbp-38h]
  __int64 v18; // [rsp+30h] [rbp-30h]

  switch ( *(_BYTE *)(a1 + 80) )
  {
    case 4:
    case 5:
      v1 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 80LL);
      break;
    case 6:
      v1 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 32LL);
      break;
    case 9:
    case 0xA:
      v1 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 56LL);
      break;
    case 0x13:
    case 0x14:
    case 0x15:
    case 0x16:
      v1 = *(_QWORD *)(a1 + 88);
      break;
    default:
      BUG();
  }
  v2 = sub_7259C0(*(unsigned __int8 *)(v1 + 264));
  v3 = *(_QWORD *)(v2 + 168);
  sub_73C230(*(_QWORD *)(*(_QWORD *)(v1 + 176) + 88LL), v2);
  v6 = v13;
  *(_QWORD *)(v2 + 168) = v3;
  v12 = v2;
  v6 &= 0xF8000000;
  v14 = 0;
  LOBYTE(v6) = v6 | 0xC1;
  v15 = 0;
  v16 = 0;
  v17 = 0;
  v18 = 0;
  LODWORD(v13) = v6;
  BYTE4(v13) = 0;
  if ( dword_4F077C4 == 2 )
    *(_BYTE *)(qword_4F04C68[0] + 776LL * unk_4F04C40 + 7) |= 8u;
  sub_5EBF70((__int64)&v12, v2, v4, v5);
  result = qword_4F04C68[0] + 776LL * unk_4F04C40;
  if ( *(_QWORD *)(result + 456) )
    result = sub_87DD20(unk_4F04C40, v2, v7, v8, v9, v10, v12, v13, v14, v15, v16, v17, v18);
  if ( dword_4F077C4 == 2 )
  {
    result = 776LL * unk_4F04C40;
    *(_BYTE *)(qword_4F04C68[0] + result + 7) &= ~8u;
    if ( *(_QWORD *)(qword_4F04C68[0] + result + 456) )
      return sub_8845B0();
  }
  return result;
}
