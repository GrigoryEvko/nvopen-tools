// Function: sub_3808940
// Address: 0x3808940
//
__int64 __fastcall sub_3808940(__int64 a1, __int64 a2)
{
  unsigned __int64 v3; // r14
  __int64 v4; // rdx
  unsigned __int64 v5; // rax
  __int64 v6; // rsi
  __int64 *v7; // rcx
  unsigned __int64 v8; // r12
  __int64 v9; // rdx
  __int64 v10; // r13
  __int64 v11; // rax
  unsigned __int16 v12; // r8
  __int64 v13; // rax
  __int64 v14; // r9
  unsigned int v15; // edx
  unsigned __int64 v16; // r8
  __int64 v17; // r10
  __int64 v18; // r11
  __int64 v19; // rax
  __int16 v20; // cx
  __int64 v21; // rax
  unsigned int v22; // esi
  __int64 v23; // r12
  bool v25; // al
  __int128 v26; // [rsp-20h] [rbp-B0h]
  __int128 v27; // [rsp-10h] [rbp-A0h]
  __int64 *v28; // [rsp+8h] [rbp-88h]
  unsigned int v29; // [rsp+8h] [rbp-88h]
  unsigned __int16 v30; // [rsp+10h] [rbp-80h]
  __int64 v31; // [rsp+10h] [rbp-80h]
  __int64 v32; // [rsp+18h] [rbp-78h]
  __int64 v33; // [rsp+20h] [rbp-70h]
  _QWORD *v34; // [rsp+28h] [rbp-68h]
  __int64 v35; // [rsp+30h] [rbp-60h]
  __int64 v36; // [rsp+40h] [rbp-50h] BYREF
  int v37; // [rsp+48h] [rbp-48h]
  __int16 v38; // [rsp+50h] [rbp-40h] BYREF
  __int64 v39; // [rsp+58h] [rbp-38h]

  v3 = sub_3805E70(a1, *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL));
  v35 = v4;
  v5 = sub_3805E70(a1, *(_QWORD *)(*(_QWORD *)(a2 + 40) + 80LL), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 88LL));
  v6 = *(_QWORD *)(a2 + 80);
  v7 = *(__int64 **)(a2 + 40);
  v8 = v5;
  v10 = v9;
  v34 = *(_QWORD **)(a1 + 8);
  v11 = *(_QWORD *)(v3 + 48) + 16LL * (unsigned int)v35;
  v12 = *(_WORD *)v11;
  v13 = *(_QWORD *)(v11 + 8);
  v36 = v6;
  v33 = v13;
  if ( v6 )
  {
    v28 = v7;
    v30 = v12;
    sub_B96E90((__int64)&v36, v6, 1);
    v7 = v28;
    v12 = v30;
  }
  v14 = v35;
  v15 = v12;
  v16 = v3;
  v37 = *(_DWORD *)(a2 + 72);
  v17 = *v7;
  v18 = v7[1];
  v19 = *(_QWORD *)(*v7 + 48) + 16LL * *((unsigned int *)v7 + 2);
  v20 = *(_WORD *)v19;
  v21 = *(_QWORD *)(v19 + 8);
  v38 = v20;
  v39 = v21;
  if ( v20 )
  {
    v22 = ((unsigned __int16)(v20 - 17) < 0xD4u) + 205;
  }
  else
  {
    v29 = v15;
    v31 = v17;
    v32 = v18;
    v25 = sub_30070B0((__int64)&v38);
    v15 = v29;
    v17 = v31;
    v18 = v32;
    v16 = v3;
    v14 = v35;
    v22 = 205 - (!v25 - 1);
  }
  *((_QWORD *)&v27 + 1) = v10;
  *(_QWORD *)&v27 = v8;
  *((_QWORD *)&v26 + 1) = v14;
  *(_QWORD *)&v26 = v16;
  v23 = sub_340EC60(v34, v22, (__int64)&v36, v15, v33, 0, v17, v18, v26, v27);
  if ( v36 )
    sub_B91220((__int64)&v36, v36);
  return v23;
}
