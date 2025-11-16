// Function: sub_1709340
// Address: 0x1709340
//
__int64 __fastcall sub_1709340(__int64 a1, __int64 a2, int a3, __int64 *a4)
{
  __int64 v7; // rax
  __int64 v8; // r12
  __int64 v9; // r15
  __int64 v10; // rdi
  __int64 *v11; // r14
  __int64 v12; // rax
  __int64 v13; // rcx
  __int64 v14; // rdx
  bool v15; // zf
  __int64 v16; // rsi
  __int64 v17; // rsi
  unsigned __int8 *v18; // rsi
  unsigned __int8 *v20; // [rsp+18h] [rbp-58h] BYREF
  __int64 v21; // [rsp+20h] [rbp-50h] BYREF
  __int16 v22; // [rsp+30h] [rbp-40h]

  v22 = 257;
  v7 = sub_1648B60(64);
  v8 = v7;
  if ( v7 )
  {
    v9 = v7;
    sub_15F1EA0(v7, a2, 53, 0, 0, 0);
    *(_DWORD *)(v8 + 56) = a3;
    sub_164B780(v8, &v21);
    sub_1648880(v8, *(_DWORD *)(v8 + 56), 1);
  }
  else
  {
    v9 = 0;
  }
  v10 = *(_QWORD *)(a1 + 8);
  if ( v10 )
  {
    v11 = *(__int64 **)(a1 + 16);
    sub_157E9D0(v10 + 40, v8);
    v12 = *(_QWORD *)(v8 + 24);
    v13 = *v11;
    *(_QWORD *)(v8 + 32) = v11;
    v13 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v8 + 24) = v13 | v12 & 7;
    *(_QWORD *)(v13 + 8) = v8 + 24;
    *v11 = *v11 & 7 | (v8 + 24);
  }
  sub_164B780(v9, a4);
  v15 = *(_QWORD *)(a1 + 80) == 0;
  v20 = (unsigned __int8 *)v8;
  if ( v15 )
    sub_4263D6(v9, a4, v14);
  (*(void (__fastcall **)(__int64, unsigned __int8 **))(a1 + 88))(a1 + 64, &v20);
  v16 = *(_QWORD *)a1;
  if ( *(_QWORD *)a1 )
  {
    v20 = *(unsigned __int8 **)a1;
    sub_1623A60((__int64)&v20, v16, 2);
    v17 = *(_QWORD *)(v8 + 48);
    if ( v17 )
      sub_161E7C0(v8 + 48, v17);
    v18 = v20;
    *(_QWORD *)(v8 + 48) = v20;
    if ( v18 )
      sub_1623210((__int64)&v20, v18, v8 + 48);
  }
  return v8;
}
