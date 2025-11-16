// Function: sub_3796290
// Address: 0x3796290
//
unsigned __int8 *__fastcall sub_3796290(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rsi
  _QWORD *v5; // r15
  __int64 v6; // rdx
  __int64 v7; // r13
  __int64 v8; // r9
  __int64 v9; // r12
  unsigned __int16 *v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  unsigned int v13; // esi
  unsigned __int8 *v14; // r12
  __int128 v16; // [rsp-20h] [rbp-80h]
  __int64 v17; // [rsp+8h] [rbp-58h]
  __int64 v18; // [rsp+10h] [rbp-50h]
  __int64 v19; // [rsp+18h] [rbp-48h]
  __int64 v20; // [rsp+20h] [rbp-40h] BYREF
  int v21; // [rsp+28h] [rbp-38h]

  v3 = sub_37946F0(a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v4 = *(_QWORD *)(a2 + 80);
  v5 = *(_QWORD **)(a1 + 8);
  v7 = v6;
  v8 = *(_QWORD *)(a2 + 40);
  v9 = v3;
  v10 = (unsigned __int16 *)(*(_QWORD *)(v3 + 48) + 16LL * (unsigned int)v6);
  v11 = *v10;
  v12 = *((_QWORD *)v10 + 1);
  v20 = v4;
  if ( v4 )
  {
    v17 = v11;
    v18 = v8;
    v19 = v12;
    sub_B96E90((__int64)&v20, v4, 1);
    v11 = v17;
    v8 = v18;
    v12 = v19;
  }
  v13 = *(_DWORD *)(a2 + 24);
  v21 = *(_DWORD *)(a2 + 72);
  *((_QWORD *)&v16 + 1) = v7;
  *(_QWORD *)&v16 = v9;
  v14 = sub_3406EB0(v5, v13, (__int64)&v20, v11, v12, v8, v16, *(_OWORD *)(v8 + 40));
  if ( v20 )
    sub_B91220((__int64)&v20, v20);
  return v14;
}
