// Function: sub_8CCC20
// Address: 0x8ccc20
//
__int64 __fastcall sub_8CCC20(__int64 a1)
{
  __int64 v2; // r13
  __int64 v3; // rdi
  char v4; // al
  __int64 v5; // rax
  __int64 v6; // rdi
  __int64 result; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r14
  _QWORD *v11; // rax

  v2 = *(_QWORD *)(a1 + 88);
  v3 = sub_878920(a1);
  v4 = *(_BYTE *)(v3 + 80);
  if ( v4 == 19 )
  {
    v3 = sub_892920(v3);
    v4 = *(_BYTE *)(v3 + 80);
  }
  switch ( v4 )
  {
    case 4:
    case 5:
      v5 = *(_QWORD *)(*(_QWORD *)(v3 + 96) + 80LL);
      break;
    case 6:
      v5 = *(_QWORD *)(*(_QWORD *)(v3 + 96) + 32LL);
      break;
    case 9:
    case 10:
      v5 = *(_QWORD *)(*(_QWORD *)(v3 + 96) + 56LL);
      break;
    case 19:
    case 20:
    case 21:
    case 22:
      v5 = *(_QWORD *)(v3 + 88);
      break;
    default:
      BUG();
  }
  v6 = *(_QWORD *)(v5 + 104);
  if ( unk_4D03FC0 && (*(_BYTE *)(v6 + 89) & 4) != 0 && !*(_QWORD *)(v6 + 32) )
    sub_8C88F0((__int64 *)v6, 0);
  result = sub_8C9880(v6);
  if ( !*(_QWORD *)(v2 + 32) )
  {
    v10 = *(_QWORD *)(*(_QWORD *)result + 88LL);
    v11 = sub_8C6880(v10, a1, v8, v9);
    if ( v11 )
      return sub_8CA500(v2, *(_QWORD *)(v11[1] + 88LL));
    else
      return sub_8CA1D0(v10, a1);
  }
  return result;
}
