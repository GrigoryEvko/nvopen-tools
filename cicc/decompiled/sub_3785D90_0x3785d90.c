// Function: sub_3785D90
// Address: 0x3785d90
//
unsigned __int8 *__fastcall sub_3785D90(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rdx
  __int128 *v5; // rax
  _QWORD *v6; // rdi
  __int64 v7; // r9
  __int64 v8; // r9
  unsigned __int8 *v9; // r14
  __int64 v10; // rdx
  __int64 v11; // r15
  _QWORD *v12; // rdi
  unsigned __int8 *v13; // r14
  __int128 v15; // [rsp-20h] [rbp-70h]
  __int128 v16; // [rsp+0h] [rbp-50h] BYREF
  __int128 v17; // [rsp+10h] [rbp-40h] BYREF
  __int64 v18; // [rsp+20h] [rbp-30h] BYREF
  int v19; // [rsp+28h] [rbp-28h]

  v3 = *(_QWORD *)(a2 + 40);
  DWORD2(v16) = 0;
  DWORD2(v17) = 0;
  *(_QWORD *)&v16 = 0;
  v4 = *(_QWORD *)(v3 + 48);
  *(_QWORD *)&v17 = 0;
  sub_375E8D0(a1, *(_QWORD *)(v3 + 40), v4, (__int64)&v16, (__int64)&v17);
  v5 = *(__int128 **)(a2 + 40);
  v6 = *(_QWORD **)(a1 + 8);
  v18 = 0;
  v19 = 0;
  v9 = sub_3406EB0(v6, 0x170u, (__int64)&v18, 1, 0, v7, *v5, v16);
  v11 = v10;
  if ( v18 )
    sub_B91220((__int64)&v18, v18);
  v12 = *(_QWORD **)(a1 + 8);
  *((_QWORD *)&v15 + 1) = v11;
  *(_QWORD *)&v15 = v9;
  v18 = 0;
  v19 = 0;
  v13 = sub_3406EB0(v12, 0x170u, (__int64)&v18, 1, 0, v8, v15, v17);
  if ( v18 )
    sub_B91220((__int64)&v18, v18);
  return v13;
}
