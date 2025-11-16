// Function: sub_37A8940
// Address: 0x37a8940
//
unsigned __int8 *__fastcall sub_37A8940(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rsi
  _QWORD *v5; // r13
  __int64 v6; // r14
  __int64 v7; // rcx
  __int64 v8; // rdx
  __int64 v9; // r15
  unsigned int v10; // r12d
  __int64 v11; // r8
  unsigned __int8 *v12; // r12
  __int128 v14; // [rsp-20h] [rbp-80h]
  __int64 v15; // [rsp+8h] [rbp-58h]
  __int64 v16; // [rsp+10h] [rbp-50h]
  __int64 v17; // [rsp+20h] [rbp-40h] BYREF
  int v18; // [rsp+28h] [rbp-38h]

  v3 = sub_379AB60(a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v4 = *(_QWORD *)(a2 + 80);
  v5 = *(_QWORD **)(a1 + 8);
  v6 = v3;
  v7 = *(_QWORD *)(a2 + 40);
  v9 = v8;
  v10 = **(unsigned __int16 **)(a2 + 48);
  v11 = *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL);
  v17 = v4;
  if ( v4 )
  {
    v15 = v7;
    v16 = v11;
    sub_B96E90((__int64)&v17, v4, 1);
    v7 = v15;
    v11 = v16;
  }
  v18 = *(_DWORD *)(a2 + 72);
  *((_QWORD *)&v14 + 1) = v9;
  *(_QWORD *)&v14 = v6;
  v12 = sub_3406EB0(v5, 0x9Eu, (__int64)&v17, v10, v11, (__int64)&v17, v14, *(_OWORD *)(v7 + 40));
  if ( v17 )
    sub_B91220((__int64)&v17, v17);
  return v12;
}
