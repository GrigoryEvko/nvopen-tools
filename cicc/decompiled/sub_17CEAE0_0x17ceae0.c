// Function: sub_17CEAE0
// Address: 0x17ceae0
//
_QWORD *__fastcall sub_17CEAE0(__int64 *a1, _QWORD *a2, __int64 a3, __int64 *a4)
{
  unsigned int v7; // edx
  _QWORD *v8; // rax
  _QWORD *v9; // r12
  __int64 v10; // rdi
  unsigned __int64 *v11; // r14
  __int64 v12; // rax
  unsigned __int64 v13; // rcx
  __int64 v14; // rsi
  __int64 v15; // rsi
  unsigned __int8 *v16; // rsi
  unsigned int v18; // [rsp+Ch] [rbp-64h]
  unsigned __int8 *v19; // [rsp+18h] [rbp-58h] BYREF
  char v20[16]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v21; // [rsp+30h] [rbp-40h]

  v7 = *(_DWORD *)(sub_1632FA0(*(_QWORD *)(*(_QWORD *)(a1[1] + 56) + 40LL)) + 4);
  v21 = 257;
  v18 = v7;
  v8 = sub_1648A60(64, 1u);
  v9 = v8;
  if ( v8 )
    sub_15F8BC0((__int64)v8, a2, v18, a3, (__int64)v20, 0);
  v10 = a1[1];
  if ( v10 )
  {
    v11 = (unsigned __int64 *)a1[2];
    sub_157E9D0(v10 + 40, (__int64)v9);
    v12 = v9[3];
    v13 = *v11;
    v9[4] = v11;
    v13 &= 0xFFFFFFFFFFFFFFF8LL;
    v9[3] = v13 | v12 & 7;
    *(_QWORD *)(v13 + 8) = v9 + 3;
    *v11 = *v11 & 7 | (unsigned __int64)(v9 + 3);
  }
  sub_164B780((__int64)v9, a4);
  v14 = *a1;
  if ( *a1 )
  {
    v19 = (unsigned __int8 *)*a1;
    sub_1623A60((__int64)&v19, v14, 2);
    v15 = v9[6];
    if ( v15 )
      sub_161E7C0((__int64)(v9 + 6), v15);
    v16 = v19;
    v9[6] = v19;
    if ( v16 )
      sub_1623210((__int64)&v19, v16, (__int64)(v9 + 6));
  }
  return v9;
}
