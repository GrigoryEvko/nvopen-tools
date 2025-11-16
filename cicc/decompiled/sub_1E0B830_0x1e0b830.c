// Function: sub_1E0B830
// Address: 0x1e0b830
//
__int64 __fastcall sub_1E0B830(__int64 a1, __int64 a2, __int64 *a3, __int64 a4)
{
  __int64 v5; // r13
  __int64 v7; // r12
  __int64 v8; // rdx
  __int64 v9; // rax

  v5 = 0;
  while ( 1 )
  {
    v7 = (__int64)sub_1E0B7C0(a1, a4);
    sub_1DD5BA0((__int64 *)(a2 + 16), v7);
    v8 = *a3;
    v9 = *(_QWORD *)v7;
    *(_QWORD *)(v7 + 8) = a3;
    v8 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)v7 = v8 | v9 & 7;
    *(_QWORD *)(v8 + 8) = v7;
    *a3 = v7 | *a3 & 7;
    if ( !v5 )
      break;
    sub_1E163F0(v7);
    if ( (*(_BYTE *)(a4 + 46) & 8) == 0 )
      return v5;
LABEL_3:
    a4 = *(_QWORD *)(a4 + 8);
  }
  v5 = v7;
  if ( (*(_BYTE *)(a4 + 46) & 8) != 0 )
    goto LABEL_3;
  return v5;
}
