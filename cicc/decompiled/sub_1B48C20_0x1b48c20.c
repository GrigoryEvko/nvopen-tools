// Function: sub_1B48C20
// Address: 0x1b48c20
//
_QWORD *__fastcall sub_1B48C20(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v10; // rax
  _QWORD *v11; // r12
  __int64 v12; // rdi
  unsigned __int64 *v13; // r13
  __int64 v14; // rax
  unsigned __int64 v15; // rcx
  __int64 v16; // rsi
  __int64 v17; // rsi
  unsigned __int8 *v18; // rsi
  unsigned __int8 *v21; // [rsp+18h] [rbp-58h] BYREF
  __int64 v22; // [rsp+20h] [rbp-50h] BYREF
  __int16 v23; // [rsp+30h] [rbp-40h]

  v23 = 257;
  v10 = sub_1648A60(56, 3u);
  v11 = v10;
  if ( v10 )
    sub_15F83E0((__int64)v10, a3, a4, a2, 0);
  if ( a5 )
    sub_1625C10((__int64)v11, 2, a5);
  if ( a6 )
    sub_1625C10((__int64)v11, 15, a6);
  v12 = a1[1];
  if ( v12 )
  {
    v13 = (unsigned __int64 *)a1[2];
    sub_157E9D0(v12 + 40, (__int64)v11);
    v14 = v11[3];
    v15 = *v13;
    v11[4] = v13;
    v15 &= 0xFFFFFFFFFFFFFFF8LL;
    v11[3] = v15 | v14 & 7;
    *(_QWORD *)(v15 + 8) = v11 + 3;
    *v13 = *v13 & 7 | (unsigned __int64)(v11 + 3);
  }
  sub_164B780((__int64)v11, &v22);
  v16 = *a1;
  if ( *a1 )
  {
    v21 = (unsigned __int8 *)*a1;
    sub_1623A60((__int64)&v21, v16, 2);
    v17 = v11[6];
    if ( v17 )
      sub_161E7C0((__int64)(v11 + 6), v17);
    v18 = v21;
    v11[6] = v21;
    if ( v18 )
      sub_1623210((__int64)&v21, v18, (__int64)(v11 + 6));
  }
  return v11;
}
