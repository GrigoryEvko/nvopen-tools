// Function: sub_156E5B0
// Address: 0x156e5b0
//
_QWORD *__fastcall sub_156E5B0(__int64 *a1, __int64 a2, __int64 a3)
{
  _QWORD *v5; // r12
  __int64 v6; // rdi
  unsigned __int64 *v7; // r13
  __int64 v8; // rax
  unsigned __int64 v9; // rcx
  __int64 v10; // rsi
  __int64 v11; // rsi
  _QWORD v13[5]; // [rsp+8h] [rbp-28h] BYREF

  v5 = (_QWORD *)sub_1648A60(64, 1);
  if ( v5 )
    sub_15F9210(v5, *(_QWORD *)(*(_QWORD *)a2 + 24LL), a2, 0, 0, 0);
  v6 = a1[1];
  if ( v6 )
  {
    v7 = (unsigned __int64 *)a1[2];
    sub_157E9D0(v6 + 40, v5);
    v8 = v5[3];
    v9 = *v7;
    v5[4] = v7;
    v9 &= 0xFFFFFFFFFFFFFFF8LL;
    v5[3] = v9 | v8 & 7;
    *(_QWORD *)(v9 + 8) = v5 + 3;
    *v7 = *v7 & 7 | (unsigned __int64)(v5 + 3);
  }
  sub_164B780(v5, a3);
  v10 = *a1;
  if ( *a1 )
  {
    v13[0] = *a1;
    sub_1623A60(v13, v10, 2);
    if ( v5[6] )
      sub_161E7C0(v5 + 6);
    v11 = v13[0];
    v5[6] = v13[0];
    if ( v11 )
      sub_1623210(v13, v11, v5 + 6);
  }
  return v5;
}
