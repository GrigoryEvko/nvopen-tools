// Function: sub_1F2B5C0
// Address: 0x1f2b5c0
//
__int64 __fastcall sub_1F2B5C0(__int64 a1, __int64 *a2, __int64 *a3, _BYTE *a4)
{
  __int64 v6; // rax
  __int64 v7; // r15
  _QWORD *v8; // r12
  __int64 v9; // rdi
  unsigned __int64 *v10; // rbx
  __int64 v11; // rax
  unsigned __int64 v12; // rcx
  __int64 v13; // rsi
  __int64 v14; // rsi
  unsigned __int8 *v15; // rsi
  __int64 v17; // rax
  unsigned __int8 *v18; // [rsp+8h] [rbp-58h] BYREF
  __int64 v19[2]; // [rsp+10h] [rbp-50h] BYREF
  __int16 v20; // [rsp+20h] [rbp-40h]

  v6 = (*(__int64 (__fastcall **)(__int64, __int64 *))(*(_QWORD *)a1 + 520LL))(a1, a3);
  if ( v6 )
  {
    v7 = v6;
    v19[0] = (__int64)"StackGuard";
    v20 = 259;
    v8 = sub_1648A60(64, 1u);
    if ( v8 )
      sub_15F9210((__int64)v8, *(_QWORD *)(*(_QWORD *)v7 + 24LL), v7, 0, 1u, 0);
    v9 = a3[1];
    if ( v9 )
    {
      v10 = (unsigned __int64 *)a3[2];
      sub_157E9D0(v9 + 40, (__int64)v8);
      v11 = v8[3];
      v12 = *v10;
      v8[4] = v10;
      v12 &= 0xFFFFFFFFFFFFFFF8LL;
      v8[3] = v12 | v11 & 7;
      *(_QWORD *)(v12 + 8) = v8 + 3;
      *v10 = *v10 & 7 | (unsigned __int64)(v8 + 3);
    }
    sub_164B780((__int64)v8, v19);
    v13 = *a3;
    if ( *a3 )
    {
      v18 = (unsigned __int8 *)*a3;
      sub_1623A60((__int64)&v18, v13, 2);
      v14 = v8[6];
      if ( v14 )
        sub_161E7C0((__int64)(v8 + 6), v14);
      v15 = v18;
      v8[6] = v18;
      if ( v15 )
        sub_1623210((__int64)&v18, v15, (__int64)(v8 + 6));
    }
  }
  else
  {
    if ( a4 )
      *a4 = 1;
    (*(void (__fastcall **)(__int64, __int64 *))(*(_QWORD *)a1 + 528LL))(a1, a2);
    v20 = 257;
    v17 = sub_15E26F0(a2, 199, 0, 0);
    return sub_1285290(a3, *(_QWORD *)(*(_QWORD *)v17 + 24LL), v17, 0, 0, (__int64)v19, 0);
  }
  return (__int64)v8;
}
