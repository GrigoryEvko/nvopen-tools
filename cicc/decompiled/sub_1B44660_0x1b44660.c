// Function: sub_1B44660
// Address: 0x1b44660
//
void __fastcall sub_1B44660(__int64 *a1, __int64 a2)
{
  _QWORD *v3; // rax
  _QWORD *v4; // rbx
  __int64 v5; // rdi
  unsigned __int64 *v6; // r13
  __int64 v7; // rax
  unsigned __int64 v8; // rcx
  __int64 v9; // rsi
  __int64 v10; // rsi
  unsigned __int8 *v11; // rsi
  unsigned __int8 *v12; // [rsp+8h] [rbp-48h] BYREF
  __int64 v13; // [rsp+10h] [rbp-40h] BYREF
  __int16 v14; // [rsp+20h] [rbp-30h]

  v14 = 257;
  v3 = sub_1648A60(56, 1u);
  v4 = v3;
  if ( v3 )
    sub_15F8320((__int64)v3, a2, 0);
  v5 = a1[1];
  if ( v5 )
  {
    v6 = (unsigned __int64 *)a1[2];
    sub_157E9D0(v5 + 40, (__int64)v4);
    v7 = v4[3];
    v8 = *v6;
    v4[4] = v6;
    v8 &= 0xFFFFFFFFFFFFFFF8LL;
    v4[3] = v8 | v7 & 7;
    *(_QWORD *)(v8 + 8) = v4 + 3;
    *v6 = *v6 & 7 | (unsigned __int64)(v4 + 3);
  }
  sub_164B780((__int64)v4, &v13);
  v9 = *a1;
  if ( *a1 )
  {
    v12 = (unsigned __int8 *)*a1;
    sub_1623A60((__int64)&v12, v9, 2);
    v10 = v4[6];
    if ( v10 )
      sub_161E7C0((__int64)(v4 + 6), v10);
    v11 = v12;
    v4[6] = v12;
    if ( v11 )
      sub_1623210((__int64)&v12, v11, (__int64)(v4 + 6));
  }
}
