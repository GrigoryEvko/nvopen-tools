// Function: sub_8C9210
// Address: 0x8c9210
//
_QWORD *__fastcall sub_8C9210(__int64 a1)
{
  __int64 v1; // r13
  __int64 v2; // rdx
  __int64 v4; // rax
  __int64 v5; // rdi
  _QWORD *result; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r14
  _QWORD *v10; // rax

  v1 = *(_QWORD *)(a1 + 88);
  if ( (*(_BYTE *)(v1 + 186) & 0x60) == 0x20 )
    return sub_8C7090(6, *(_QWORD *)(a1 + 88));
  v2 = **(_QWORD **)(*(_QWORD *)(v1 + 168) + 16LL);
  switch ( *(_BYTE *)(v2 + 80) )
  {
    case 4:
    case 5:
      v4 = *(_QWORD *)(*(_QWORD *)(v2 + 96) + 80LL);
      break;
    case 6:
      v4 = *(_QWORD *)(*(_QWORD *)(v2 + 96) + 32LL);
      break;
    case 9:
    case 0xA:
      v4 = *(_QWORD *)(*(_QWORD *)(v2 + 96) + 56LL);
      break;
    case 0x13:
    case 0x14:
    case 0x15:
    case 0x16:
      v4 = *(_QWORD *)(v2 + 88);
      break;
    default:
      BUG();
  }
  v5 = *(_QWORD *)(v4 + 104);
  if ( unk_4D03FC0 && (*(_BYTE *)(v5 + 89) & 4) != 0 && !*(_QWORD *)(v5 + 32) )
    sub_8C88F0((__int64 *)v5, 0);
  result = (_QWORD *)sub_8C9880(v5);
  if ( !*(_QWORD *)(v1 + 32) )
  {
    v9 = *(_QWORD *)(*result + 88LL);
    v10 = sub_8C6880(v9, a1, v7, v8);
    if ( v10 )
      return (_QWORD *)sub_8CA500(v1, *(_QWORD *)(v10[1] + 88LL));
    else
      return (_QWORD *)sub_8CA1D0(v9, a1);
  }
  return result;
}
