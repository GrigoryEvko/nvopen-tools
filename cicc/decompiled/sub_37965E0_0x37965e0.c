// Function: sub_37965E0
// Address: 0x37965e0
//
unsigned __int8 *__fastcall sub_37965E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int16 *v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rdx
  unsigned __int64 *v9; // rdx
  unsigned __int64 v10; // rsi
  int v11; // eax
  __int64 v12; // rsi
  unsigned __int64 v13; // r11
  __int64 v14; // rax
  _QWORD *v15; // r15
  __int64 v16; // rdx
  __int64 v17; // r13
  __int64 v18; // r12
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  unsigned int v24; // esi
  unsigned __int8 *v25; // r12
  __int64 v27; // rdx
  unsigned __int64 v28; // rdx
  __int128 v29; // [rsp-20h] [rbp-90h]
  __int128 v30; // [rsp-10h] [rbp-80h]
  __int64 v31; // [rsp+8h] [rbp-68h]
  unsigned int v32; // [rsp+10h] [rbp-60h]
  __int64 v33; // [rsp+10h] [rbp-60h]
  __int64 v34; // [rsp+10h] [rbp-60h]
  __int64 v35; // [rsp+10h] [rbp-60h]
  __int64 v36; // [rsp+18h] [rbp-58h]
  unsigned __int64 v37; // [rsp+20h] [rbp-50h]
  __int64 v38; // [rsp+28h] [rbp-48h]
  __int64 v39; // [rsp+30h] [rbp-40h] BYREF
  __int64 v40; // [rsp+38h] [rbp-38h]

  v6 = *(unsigned __int16 **)(a2 + 48);
  LODWORD(v7) = *v6;
  v8 = *((_QWORD *)v6 + 1);
  LOWORD(v39) = v7;
  v40 = v8;
  if ( (_WORD)v7 )
  {
    v38 = 0;
    LOWORD(v7) = word_4456580[(int)v7 - 1];
  }
  else
  {
    v34 = a5;
    v7 = sub_3009970((__int64)&v39, a2, v8, a4, a5);
    a5 = v34;
    v38 = v27;
    a4 = v7;
  }
  v9 = *(unsigned __int64 **)(a2 + 40);
  LOWORD(a4) = v7;
  v10 = v9[5];
  v11 = *(unsigned __int16 *)(v10 + 96);
  v12 = *(_QWORD *)(v10 + 104);
  LOWORD(v39) = v11;
  v40 = v12;
  if ( (_WORD)v11 )
  {
    v13 = 0;
    LOWORD(v11) = word_4456580[v11 - 1];
  }
  else
  {
    v35 = a4;
    v11 = sub_3009970((__int64)&v39, v12, (__int64)v9, a4, a5);
    a4 = v35;
    v13 = v28;
    WORD1(a5) = HIWORD(v11);
    v9 = *(unsigned __int64 **)(a2 + 40);
  }
  LOWORD(a5) = v11;
  v31 = a4;
  v37 = v13;
  v32 = a5;
  v14 = sub_37946F0(a1, *v9, v9[1]);
  v15 = *(_QWORD **)(a1 + 8);
  v17 = v16;
  v18 = v14;
  v19 = sub_33F7D60(v15, v32, v37);
  v21 = v31;
  v22 = v19;
  v23 = v20;
  v39 = *(_QWORD *)(a2 + 80);
  if ( v39 )
  {
    v36 = v20;
    v33 = v19;
    sub_B96E90((__int64)&v39, v39, 1);
    v21 = v31;
    v22 = v33;
    v23 = v36;
  }
  *((_QWORD *)&v30 + 1) = v23;
  *(_QWORD *)&v30 = v22;
  v24 = *(_DWORD *)(a2 + 24);
  *((_QWORD *)&v29 + 1) = v17;
  *(_QWORD *)&v29 = v18;
  LODWORD(v40) = *(_DWORD *)(a2 + 72);
  v25 = sub_3406EB0(v15, v24, (__int64)&v39, v21, v38, v23, v29, v30);
  if ( v39 )
    sub_B91220((__int64)&v39, v39);
  return v25;
}
