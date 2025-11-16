// Function: sub_3796780
// Address: 0x3796780
//
unsigned __int8 *__fastcall sub_3796780(__int64 *a1, __int64 a2, __m128i a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r10d
  unsigned int v7; // r14d
  __int64 v9; // rsi
  __int64 *v10; // rax
  __int64 v11; // rcx
  unsigned __int64 v12; // r12
  __int64 v13; // r13
  unsigned __int16 *v14; // rdx
  int v15; // eax
  __int64 v16; // rdx
  unsigned __int16 *v17; // rdx
  int v18; // eax
  __int64 v19; // rdx
  __int128 v20; // rax
  __int64 v21; // r9
  int v22; // r9d
  int v23; // eax
  unsigned __int8 *v24; // r12
  __int64 v26; // rdx
  __int64 v27; // rdx
  __int128 v28; // [rsp-20h] [rbp-C0h]
  unsigned int v29; // [rsp+8h] [rbp-98h]
  unsigned int v30; // [rsp+10h] [rbp-90h]
  __int64 v31; // [rsp+18h] [rbp-88h]
  __int64 v32; // [rsp+20h] [rbp-80h]
  unsigned int v33; // [rsp+28h] [rbp-78h]
  _QWORD *v34; // [rsp+28h] [rbp-78h]
  __int16 v35; // [rsp+2Ah] [rbp-76h]
  __int64 v36; // [rsp+30h] [rbp-70h] BYREF
  int v37; // [rsp+38h] [rbp-68h]
  unsigned __int16 v38; // [rsp+40h] [rbp-60h] BYREF
  __int64 v39; // [rsp+48h] [rbp-58h]
  __int16 v40; // [rsp+50h] [rbp-50h] BYREF
  __int64 v41; // [rsp+58h] [rbp-48h]

  v9 = *(_QWORD *)(a2 + 80);
  v36 = v9;
  if ( v9 )
  {
    v35 = HIWORD(v6);
    sub_B96E90((__int64)&v36, v9, 1);
    HIWORD(v6) = v35;
  }
  v37 = *(_DWORD *)(a2 + 72);
  v10 = *(__int64 **)(a2 + 40);
  v11 = *v10;
  v12 = *v10;
  v13 = v10[1];
  v14 = (unsigned __int16 *)(*(_QWORD *)(*v10 + 48) + 16LL * *((unsigned int *)v10 + 2));
  v15 = *v14;
  v16 = *((_QWORD *)v14 + 1);
  v38 = v15;
  v39 = v16;
  if ( (_WORD)v15 )
  {
    v31 = 0;
    LOWORD(v15) = word_4456580[v15 - 1];
  }
  else
  {
    v15 = sub_3009970((__int64)&v38, v9, v16, v11, a6);
    v31 = v27;
    HIWORD(v6) = HIWORD(v15);
  }
  v17 = *(unsigned __int16 **)(a2 + 48);
  LOWORD(v6) = v15;
  v18 = *v17;
  v19 = *((_QWORD *)v17 + 1);
  v40 = v18;
  v41 = v19;
  if ( (_WORD)v18 )
  {
    v32 = 0;
    LOWORD(v18) = word_4456580[v18 - 1];
  }
  else
  {
    v30 = v6;
    v18 = sub_3009970((__int64)&v40, v9, v19, v11, a6);
    v6 = v30;
    v32 = v26;
    HIWORD(v7) = HIWORD(v18);
  }
  LOWORD(v7) = v18;
  v33 = v6;
  sub_2FE6CC0((__int64)&v40, *a1, *(_QWORD *)(a1[1] + 64), v38, v39);
  if ( (_BYTE)v40 != 5 )
  {
    v29 = v33;
    v34 = (_QWORD *)a1[1];
    *(_QWORD *)&v20 = sub_3400EE0((__int64)v34, 0, (__int64)&v36, 0, a3);
    *((_QWORD *)&v28 + 1) = v13;
    *(_QWORD *)&v28 = v12;
    sub_3406EB0(v34, 0x9Eu, (__int64)&v36, v29, v31, v21, v28, v20);
    v23 = *(_DWORD *)(a2 + 24);
    if ( v23 != 224 )
      goto LABEL_9;
LABEL_17:
    v24 = sub_33FAF80(a1[1], 213, (__int64)&v36, v7, v32, v22, a3);
    goto LABEL_11;
  }
  sub_37946F0((__int64)a1, v12, v13);
  v23 = *(_DWORD *)(a2 + 24);
  if ( v23 == 224 )
    goto LABEL_17;
LABEL_9:
  if ( v23 == 225 )
  {
    v24 = sub_33FAF80(a1[1], 214, (__int64)&v36, v7, v32, v22, a3);
  }
  else
  {
    if ( v23 != 223 )
      BUG();
    v24 = sub_33FAF80(a1[1], 215, (__int64)&v36, v7, v32, v22, a3);
  }
LABEL_11:
  if ( v36 )
    sub_B91220((__int64)&v36, v36);
  return v24;
}
