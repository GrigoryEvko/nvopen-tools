// Function: sub_8933F0
// Address: 0x8933f0
//
__int64 __fastcall sub_8933F0(__int64 a1, __int64 a2, FILE *a3)
{
  __int64 v3; // rax
  unsigned __int64 v4; // rcx
  __int64 result; // rax

  switch ( *(_BYTE *)(a1 + 80) )
  {
    case 4:
    case 5:
      v3 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 80LL);
      break;
    case 6:
      v3 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 32LL);
      break;
    case 9:
    case 0xA:
      v3 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 56LL);
      break;
    case 0x13:
    case 0x14:
    case 0x15:
    case 0x16:
      v3 = *(_QWORD *)(a1 + 88);
      break;
    default:
      BUG();
  }
  v4 = *(unsigned int *)(v3 + 40);
  result = 0;
  if ( v4 >= unk_4D042F0 )
  {
    sub_6854C0(0x1C8u, a3, a2);
    return 1;
  }
  return result;
}
