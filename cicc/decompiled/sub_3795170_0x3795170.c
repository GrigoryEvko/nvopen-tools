// Function: sub_3795170
// Address: 0x3795170
//
unsigned __int8 *__fastcall sub_3795170(__int64 *a1, unsigned __int64 a2, int a3, __m128i a4)
{
  __int64 v5; // rax
  __int64 v6; // rsi
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  unsigned __int16 v10; // bx
  __int64 v11; // rcx
  _QWORD *v12; // r14
  __int64 v13; // rax
  __int16 v14; // bx
  __int64 v15; // rax
  unsigned int v16; // esi
  __int64 v17; // rbx
  unsigned __int8 *v18; // r14
  unsigned __int16 *v19; // rdx
  int v20; // r9d
  unsigned __int8 *v21; // rax
  __int64 v22; // rdx
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // rdx
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // rdx
  __int128 v30; // [rsp-10h] [rbp-D0h]
  unsigned __int16 v31; // [rsp+0h] [rbp-C0h]
  __int64 v32; // [rsp+10h] [rbp-B0h]
  _QWORD v34[2]; // [rsp+30h] [rbp-90h] BYREF
  unsigned __int16 v35; // [rsp+40h] [rbp-80h] BYREF
  __int64 v36; // [rsp+48h] [rbp-78h]
  unsigned __int16 v37; // [rsp+50h] [rbp-70h] BYREF
  __int64 v38; // [rsp+58h] [rbp-68h]
  __int64 v39; // [rsp+60h] [rbp-60h] BYREF
  int v40; // [rsp+68h] [rbp-58h]
  unsigned __int16 v41; // [rsp+70h] [rbp-50h] BYREF
  __int64 v42; // [rsp+78h] [rbp-48h]
  __int16 v43; // [rsp+80h] [rbp-40h]
  __int64 v44; // [rsp+88h] [rbp-38h]

  v5 = sub_37946F0((__int64)a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v6 = *(_QWORD *)(a2 + 80);
  v34[0] = v5;
  v7 = *(_QWORD *)(a2 + 48);
  v34[1] = v8;
  v9 = *(_QWORD *)(v7 + 8);
  v10 = *(_WORD *)v7;
  v39 = v6;
  v36 = v9;
  v11 = *(_QWORD *)(v7 + 24);
  LOWORD(v7) = *(_WORD *)(v7 + 16);
  v35 = v10;
  v38 = v11;
  v37 = v7;
  if ( v6 )
  {
    sub_B96E90((__int64)&v39, v6, 1);
    v10 = v35;
  }
  v12 = (_QWORD *)a1[1];
  v40 = *(_DWORD *)(a2 + 72);
  if ( v10 )
  {
    if ( (unsigned __int16)(v10 - 17) > 0xD3u )
    {
LABEL_5:
      v13 = v36;
      goto LABEL_6;
    }
    v10 = word_4456580[v10 - 1];
    v13 = 0;
  }
  else
  {
    if ( !sub_30070B0((__int64)&v35) )
      goto LABEL_5;
    v10 = sub_3009970((__int64)&v35, v6, v27, v28, (__int64)v34);
    v13 = v29;
  }
LABEL_6:
  v41 = v10;
  v14 = v37;
  v42 = v13;
  if ( v37 )
  {
    if ( (unsigned __int16)(v37 - 17) > 0xD3u )
    {
LABEL_8:
      v15 = v38;
      goto LABEL_9;
    }
    v14 = word_4456580[v37 - 1];
    v15 = 0;
  }
  else
  {
    if ( !sub_30070B0((__int64)&v37) )
      goto LABEL_8;
    v14 = sub_3009970((__int64)&v37, v6, v24, v25, (__int64)v34);
    v15 = v26;
  }
LABEL_9:
  v16 = *(_DWORD *)(a2 + 24);
  *((_QWORD *)&v30 + 1) = 1;
  *(_QWORD *)&v30 = v34;
  v43 = v14;
  v44 = v15;
  v17 = (unsigned int)(1 - a3);
  v18 = sub_3411BE0(v12, v16, (__int64)&v39, &v41, 2, 1, v30);
  v19 = (unsigned __int16 *)(*(_QWORD *)(a2 + 48) + 16 * v17);
  v31 = *v19;
  v32 = *((_QWORD *)v19 + 1);
  sub_2FE6CC0((__int64)&v41, *a1, *(_QWORD *)(a1[1] + 64), *v19, v32);
  if ( (_BYTE)v41 == 5 )
  {
    sub_375FC90((__int64)a1, a2, v17, (unsigned __int64)v18, v17);
  }
  else
  {
    v21 = sub_33FAF80(a1[1], 167, (__int64)&v39, v31, v32, v20, a4);
    sub_3760E70((__int64)a1, a2, v17, (unsigned __int64)v21, v22);
  }
  if ( v39 )
    sub_B91220((__int64)&v39, v39);
  return v18;
}
