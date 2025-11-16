// Function: sub_945FB0
// Address: 0x945fb0
//
__int64 __fastcall sub_945FB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  _DWORD *v4; // r13
  __int64 v5; // rsi
  __int64 v6; // rcx
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r12

  v4 = (_DWORD *)(a2 + 64);
  sub_945E50(a1, (_DWORD *)(a2 + 64), a3, a4);
  v5 = sub_9380F0((_QWORD **)(*(_QWORD *)(a1 + 32) + 8LL), a2, 0);
  sub_938F90(a1, v5, v4, v6);
  if ( !dword_4D04658 )
  {
    v5 = a1 + 48;
    sub_93FF00(*(_QWORD **)(*(_QWORD *)(a1 + 32) + 368LL), a1 + 48);
  }
  v9 = *(_QWORD *)(a1 + 456);
  if ( v9 )
  {
    if ( v9 != -8192 && v9 != -4096 )
      sub_BD60C0(a1 + 440);
    *(_QWORD *)(a1 + 456) = 0;
  }
  return sub_B43D60(v9, v5, v7, v8);
}
