// Function: sub_37977B0
// Address: 0x37977b0
//
__int64 __fastcall sub_37977B0(__int64 a1, __int64 a2)
{
  __int64 v3; // r14
  __int64 v4; // rdx
  unsigned int v5; // r15d
  __int64 v6; // rax
  __int64 v7; // rsi
  __int64 *v8; // rcx
  __int64 v9; // r12
  __int64 v10; // rdx
  __int64 v11; // r13
  __int64 v12; // rax
  unsigned __int16 v13; // r8
  __int64 v14; // rax
  __int64 v15; // r9
  unsigned int v16; // edx
  __int64 v17; // r8
  __int64 v18; // r10
  __int64 v19; // r11
  __int64 v20; // rax
  __int16 v21; // cx
  __int64 v22; // rax
  unsigned int v23; // esi
  __int64 v24; // r12
  bool v26; // al
  __int128 v27; // [rsp-20h] [rbp-B0h]
  __int128 v28; // [rsp-10h] [rbp-A0h]
  __int64 *v29; // [rsp+8h] [rbp-88h]
  unsigned int v30; // [rsp+8h] [rbp-88h]
  unsigned __int16 v31; // [rsp+10h] [rbp-80h]
  __int64 v32; // [rsp+10h] [rbp-80h]
  __int64 v33; // [rsp+18h] [rbp-78h]
  __int64 v34; // [rsp+20h] [rbp-70h]
  __int64 v35; // [rsp+30h] [rbp-60h]
  _QWORD *v36; // [rsp+38h] [rbp-58h]
  __int64 v37; // [rsp+40h] [rbp-50h] BYREF
  int v38; // [rsp+48h] [rbp-48h]
  __int16 v39; // [rsp+50h] [rbp-40h] BYREF
  __int64 v40; // [rsp+58h] [rbp-38h]

  v3 = sub_37946F0(a1, *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL));
  v5 = v4;
  v34 = v4;
  v36 = *(_QWORD **)(a1 + 8);
  v6 = sub_37946F0(a1, *(_QWORD *)(*(_QWORD *)(a2 + 40) + 80LL), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 88LL));
  v7 = *(_QWORD *)(a2 + 80);
  v8 = *(__int64 **)(a2 + 40);
  v9 = v6;
  v11 = v10;
  v12 = *(_QWORD *)(v3 + 48) + 16LL * v5;
  v13 = *(_WORD *)v12;
  v14 = *(_QWORD *)(v12 + 8);
  v37 = v7;
  v35 = v14;
  if ( v7 )
  {
    v29 = v8;
    v31 = v13;
    sub_B96E90((__int64)&v37, v7, 1);
    v8 = v29;
    v13 = v31;
  }
  v15 = v34;
  v16 = v13;
  v17 = v3;
  v38 = *(_DWORD *)(a2 + 72);
  v18 = *v8;
  v19 = v8[1];
  v20 = *(_QWORD *)(*v8 + 48) + 16LL * *((unsigned int *)v8 + 2);
  v21 = *(_WORD *)v20;
  v22 = *(_QWORD *)(v20 + 8);
  v39 = v21;
  v40 = v22;
  if ( v21 )
  {
    v23 = ((unsigned __int16)(v21 - 17) < 0xD4u) + 205;
  }
  else
  {
    v30 = v16;
    v32 = v18;
    v33 = v19;
    v26 = sub_30070B0((__int64)&v39);
    v16 = v30;
    v18 = v32;
    v19 = v33;
    v17 = v3;
    v15 = v34;
    v23 = 205 - (!v26 - 1);
  }
  *((_QWORD *)&v28 + 1) = v11;
  *(_QWORD *)&v28 = v9;
  *((_QWORD *)&v27 + 1) = v15;
  *(_QWORD *)&v27 = v17;
  v24 = sub_340EC60(v36, v23, (__int64)&v37, v16, v35, 0, v18, v19, v27, v28);
  if ( v37 )
    sub_B91220((__int64)&v37, v37);
  return v24;
}
