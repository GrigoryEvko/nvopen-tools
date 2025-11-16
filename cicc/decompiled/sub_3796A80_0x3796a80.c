// Function: sub_3796A80
// Address: 0x3796a80
//
_QWORD *__fastcall sub_3796A80(__int64 *a1, __int64 a2, __m128i a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int16 *v7; // rdx
  int v8; // eax
  __int64 v9; // rdx
  unsigned int v10; // ecx
  __int64 v11; // rsi
  unsigned __int64 *v12; // rax
  unsigned __int64 v13; // r12
  __int64 v14; // r13
  __int64 v15; // rax
  unsigned __int16 v16; // dx
  __int64 v17; // r8
  __int64 v18; // rsi
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r8
  int v23; // eax
  unsigned int v24; // ecx
  __int128 v25; // rax
  __int64 v26; // r9
  unsigned __int64 v27; // rax
  unsigned int v28; // edx
  _QWORD *v29; // r12
  __int64 v31; // rdx
  __int64 v32; // rdx
  __int128 v33; // [rsp-20h] [rbp-C0h]
  __int64 v34; // [rsp+8h] [rbp-98h]
  _QWORD *v35; // [rsp+10h] [rbp-90h]
  __int64 v36; // [rsp+18h] [rbp-88h]
  unsigned int v37; // [rsp+20h] [rbp-80h]
  __int16 v38; // [rsp+22h] [rbp-7Eh]
  unsigned int v39; // [rsp+28h] [rbp-78h]
  __int16 v40; // [rsp+2Ah] [rbp-76h]
  unsigned __int16 v41; // [rsp+30h] [rbp-70h] BYREF
  __int64 v42; // [rsp+38h] [rbp-68h]
  __int64 v43; // [rsp+40h] [rbp-60h] BYREF
  int v44; // [rsp+48h] [rbp-58h]
  __int16 v45; // [rsp+50h] [rbp-50h] BYREF
  __int64 v46; // [rsp+58h] [rbp-48h]

  v7 = *(unsigned __int16 **)(a2 + 48);
  v8 = *v7;
  v9 = *((_QWORD *)v7 + 1);
  v45 = v8;
  v46 = v9;
  if ( (_WORD)v8 )
  {
    v36 = 0;
    LOWORD(v8) = word_4456580[v8 - 1];
  }
  else
  {
    v8 = sub_3009970((__int64)&v45, a2, v9, a5, a6);
    v40 = HIWORD(v8);
    v36 = v32;
  }
  HIWORD(v10) = v40;
  v11 = *(_QWORD *)(a2 + 80);
  LOWORD(v10) = v8;
  v12 = *(unsigned __int64 **)(a2 + 40);
  v39 = v10;
  v13 = *v12;
  v14 = v12[1];
  v15 = *(_QWORD *)(*v12 + 48) + 16LL * *((unsigned int *)v12 + 2);
  v16 = *(_WORD *)v15;
  v17 = *(_QWORD *)(v15 + 8);
  v43 = v11;
  v41 = v16;
  v42 = v17;
  if ( v11 )
  {
    sub_B96E90((__int64)&v43, v11, 1);
    v16 = v41;
    v17 = v42;
  }
  v18 = *a1;
  v44 = *(_DWORD *)(a2 + 72);
  sub_2FE6CC0((__int64)&v45, v18, *(_QWORD *)(a1[1] + 64), v16, v17);
  if ( (_BYTE)v45 == 5 )
  {
    v27 = sub_37946F0((__int64)a1, v13, v14);
  }
  else
  {
    if ( v41 )
    {
      v22 = 0;
      LOWORD(v23) = word_4456580[v41 - 1];
    }
    else
    {
      v23 = sub_3009970((__int64)&v41, v18, v19, v20, v21);
      v38 = HIWORD(v23);
      v22 = v31;
    }
    HIWORD(v24) = v38;
    v34 = v22;
    LOWORD(v24) = v23;
    v35 = (_QWORD *)a1[1];
    v37 = v24;
    *(_QWORD *)&v25 = sub_3400EE0((__int64)v35, 0, (__int64)&v43, 0, a3);
    *((_QWORD *)&v33 + 1) = v14;
    *(_QWORD *)&v33 = v13;
    v27 = (unsigned __int64)sub_3406EB0(v35, 0x9Eu, (__int64)&v43, v37, v34, v26, v33, v25);
  }
  v29 = sub_33F2D30(
          (_QWORD *)a1[1],
          (__int64)&v43,
          v39,
          v36,
          v27,
          v28 | v14 & 0xFFFFFFFF00000000LL,
          *(_DWORD *)(a2 + 96),
          *(_DWORD *)(a2 + 100));
  if ( v43 )
    sub_B91220((__int64)&v43, v43);
  return v29;
}
