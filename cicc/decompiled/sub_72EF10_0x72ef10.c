// Function: sub_72EF10
// Address: 0x72ef10
//
__int64 __fastcall sub_72EF10(__int64 a1)
{
  _QWORD *v1; // rax
  __int64 v2; // r14
  __int64 v3; // r12
  __int64 v4; // r15
  __int64 v5; // rax
  __int64 v6; // r13
  __int64 v7; // r13
  __int64 v8; // rdx

  v1 = sub_7259C0(14);
  v2 = v1[21];
  v3 = (__int64)v1;
  switch ( *(_BYTE *)(a1 + 80) )
  {
    case 4:
    case 5:
      v4 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 80LL);
      break;
    case 6:
      v4 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 32LL);
      break;
    case 9:
    case 0xA:
      v4 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 56LL);
      break;
    case 0x13:
    case 0x14:
    case 0x15:
    case 0x16:
      v4 = *(_QWORD *)(a1 + 88);
      break;
    default:
      v4 = 0;
      break;
  }
  v5 = sub_87EBB0(3, *(_QWORD *)a1);
  *(_QWORD *)(v5 + 88) = v3;
  v6 = v5;
  *(_DWORD *)(v2 + 28) = -2;
  *(_QWORD *)(v2 + 32) = a1;
  sub_8D6090(v3);
  sub_877D80(v3, v6);
  if ( (*(_BYTE *)(a1 + 81) & 0x20) != 0 )
  {
LABEL_6:
    v7 = *(_QWORD *)(v4 + 104);
    if ( !v7 )
      return v3;
    goto LABEL_9;
  }
  v7 = *(_QWORD *)(v4 + 104);
  v8 = *(_QWORD *)(v7 + 40);
  if ( v8 )
  {
    sub_72EE40(v3, 6u, v8);
    *(_BYTE *)(v3 + 89) = *(_BYTE *)(*(_QWORD *)(v4 + 104) + 89LL) & 4 | *(_BYTE *)(v3 + 89) & 0xFB;
    goto LABEL_6;
  }
LABEL_9:
  if ( (unsigned int)sub_825090(v7) )
  {
    sub_8250A0(v7);
  }
  else if ( (unsigned int)sub_8250B0(v7) )
  {
    sub_8250C0(v7);
  }
  return v3;
}
