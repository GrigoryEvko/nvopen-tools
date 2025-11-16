// Function: sub_15E7150
// Address: 0x15e7150
//
_QWORD *__fastcall sub_15E7150(__int64 *a1, _QWORD *a2)
{
  _QWORD *v2; // r12
  __int64 v3; // rbx
  __int64 v5; // r14
  __int64 v6; // rax
  _QWORD *v7; // rbx
  unsigned __int64 *v8; // r12
  unsigned __int64 v9; // rcx
  __int64 v10; // rax
  __int64 v11; // rsi
  __int64 v12; // rsi
  _QWORD v13[2]; // [rsp+0h] [rbp-40h] BYREF
  __int16 v14; // [rsp+10h] [rbp-30h]

  v2 = a2;
  v3 = *a2;
  if ( (unsigned __int8)sub_1642F90(*(_QWORD *)(*a2 + 24LL), 8) )
    return v2;
  v5 = sub_16471D0(a1[3], *(_DWORD *)(v3 + 8) >> 8);
  v14 = 257;
  v6 = sub_1648A60(56, 1);
  v7 = (_QWORD *)v6;
  if ( v6 )
    sub_15FD590(v6, a2, v5, v13, 0);
  v8 = (unsigned __int64 *)a1[2];
  sub_157E9D0(a1[1] + 40, (__int64)v7);
  v9 = *v8;
  v10 = v7[3];
  v7[4] = v8;
  v9 &= 0xFFFFFFFFFFFFFFF8LL;
  v7[3] = v9 | v10 & 7;
  *(_QWORD *)(v9 + 8) = v7 + 3;
  *v8 = *v8 & 7 | (unsigned __int64)(v7 + 3);
  v11 = *a1;
  v2 = v7;
  if ( !*a1 )
    return v2;
  v13[0] = *a1;
  sub_1623A60(v13, v11, 2);
  if ( v7[6] )
    sub_161E7C0(v7 + 6);
  v12 = v13[0];
  v7[6] = v13[0];
  if ( v12 )
    sub_1623210(v13, v12, v7 + 6);
  return v7;
}
