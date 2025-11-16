// Function: sub_2E7E840
// Address: 0x2e7e840
//
__int64 __fastcall sub_2E7E840(_QWORD *a1, __int64 a2, __int64 *a3, unsigned __int64 a4)
{
  __int64 v4; // r15
  __int64 v5; // r13
  __int64 v7; // r12
  __int64 v8; // rsi
  __int64 v9; // rax
  __int64 *v12; // [rsp+8h] [rbp-38h]

  v4 = a4;
  v5 = 0;
  v12 = (__int64 *)(a2 + 40);
  while ( 1 )
  {
    v7 = (__int64)sub_2E7B2C0(a1, v4);
    sub_2E31040(v12, v7);
    v8 = *a3;
    v9 = *(_QWORD *)v7;
    *(_QWORD *)(v7 + 8) = a3;
    v8 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)v7 = v8 | v9 & 7;
    *(_QWORD *)(v8 + 8) = v7;
    *a3 = v7 | *a3 & 7;
    if ( !v5 )
      break;
    sub_2E89030(v7);
    if ( (*(_BYTE *)(v4 + 44) & 8) == 0 )
      goto LABEL_6;
LABEL_3:
    v4 = *(_QWORD *)(v4 + 8);
  }
  v5 = v7;
  if ( (*(_BYTE *)(v4 + 44) & 8) != 0 )
    goto LABEL_3;
LABEL_6:
  if ( (unsigned __int8)sub_2E88F60(a4) )
    sub_2E7E170((__int64)a1, a4, v5);
  return v5;
}
