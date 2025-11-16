// Function: sub_3799300
// Address: 0x3799300
//
__int64 __fastcall sub_3799300(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rsi
  _QWORD *v5; // r12
  __int64 v6; // r14
  __int64 v7; // rax
  __int64 v8; // rcx
  __int64 v9; // rdx
  __int64 v10; // r15
  unsigned int v11; // r13d
  __int64 v12; // r8
  __int64 v13; // r12
  __int128 v15; // [rsp-30h] [rbp-90h]
  __int64 v16; // [rsp+8h] [rbp-58h]
  __int64 v17; // [rsp+10h] [rbp-50h]
  __int64 v18; // [rsp+20h] [rbp-40h] BYREF
  int v19; // [rsp+28h] [rbp-38h]

  HIWORD(v11) = 0;
  v3 = sub_37946F0(a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v4 = *(_QWORD *)(a2 + 80);
  v5 = *(_QWORD **)(a1 + 8);
  v6 = v3;
  v7 = *(_QWORD *)(a2 + 48);
  v8 = *(_QWORD *)(a2 + 40);
  v10 = v9;
  LOWORD(v11) = *(_WORD *)v7;
  v12 = *(_QWORD *)(v7 + 8);
  v18 = v4;
  if ( v4 )
  {
    v16 = v12;
    v17 = v8;
    sub_B96E90((__int64)&v18, v4, 1);
    v12 = v16;
    v8 = v17;
  }
  v19 = *(_DWORD *)(a2 + 72);
  *((_QWORD *)&v15 + 1) = v10;
  *(_QWORD *)&v15 = v6;
  v13 = sub_340F900(v5, 0xCDu, (__int64)&v18, v11, v12, (__int64)&v18, v15, *(_OWORD *)(v8 + 40), *(_OWORD *)(v8 + 80));
  if ( v18 )
    sub_B91220((__int64)&v18, v18);
  return v13;
}
