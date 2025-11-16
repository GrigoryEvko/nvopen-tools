// Function: sub_15E6C70
// Address: 0x15e6c70
//
_QWORD *__fastcall sub_15E6C70(__int64 a1, int a2, int a3, __int64 a4, __int64 a5, __int64 *a6, __int64 a7)
{
  __int64 v8; // r13
  _QWORD *v9; // r12
  unsigned __int64 *v10; // r13
  unsigned __int64 v11; // rcx
  __int64 v12; // rax
  __int64 v13; // rsi
  __int64 v14; // rsi
  unsigned int v17; // [rsp+14h] [rbp-5Ch]
  _QWORD v20[7]; // [rsp+38h] [rbp-38h] BYREF

  v8 = *(_QWORD *)(*(_QWORD *)a1 + 24LL);
  v17 = a5 + 3;
  v9 = (_QWORD *)sub_1648AB0(72, (unsigned int)(a5 + 3), 0);
  if ( v9 )
  {
    sub_15F1EA0(v9, **(_QWORD **)(v8 + 16), 5, &v9[-3 * v17], v17, 0);
    v9[7] = 0;
    sub_15F6500((_DWORD)v9, v8, a1, a2, a3, a7, a4, a5, 0, 0);
  }
  v10 = (unsigned __int64 *)a6[2];
  sub_157E9D0(a6[1] + 40, (__int64)v9);
  v11 = *v10;
  v12 = v9[3];
  v9[4] = v10;
  v11 &= 0xFFFFFFFFFFFFFFF8LL;
  v9[3] = v11 | v12 & 7;
  *(_QWORD *)(v11 + 8) = v9 + 3;
  *v10 = *v10 & 7 | (unsigned __int64)(v9 + 3);
  v13 = *a6;
  if ( *a6 )
  {
    v20[0] = *a6;
    sub_1623A60(v20, v13, 2);
    if ( v9[6] )
      sub_161E7C0(v9 + 6);
    v14 = v20[0];
    v9[6] = v20[0];
    if ( v14 )
      sub_1623210(v20, v14, v9 + 6);
  }
  return v9;
}
