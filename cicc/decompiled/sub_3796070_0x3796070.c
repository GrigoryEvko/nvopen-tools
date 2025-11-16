// Function: sub_3796070
// Address: 0x3796070
//
unsigned __int8 *__fastcall sub_3796070(__int64 *a1, __int64 a2, __m128i a3)
{
  __int64 v4; // rsi
  __int64 v5; // rsi
  unsigned __int64 *v6; // rax
  unsigned __int64 v7; // r12
  __int64 v8; // r13
  __int64 v9; // rax
  unsigned __int16 v10; // dx
  __int64 v11; // r8
  __int64 v12; // rax
  __int64 v13; // r11
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r8
  int v18; // eax
  unsigned int v19; // ecx
  __int128 v20; // rax
  __int64 v21; // r9
  __int64 v22; // rsi
  __int64 v23; // r8
  __int64 v24; // r9
  unsigned int v25; // edx
  __int64 v26; // rcx
  unsigned __int16 *v27; // rdx
  _QWORD *v28; // r15
  __int64 v29; // r10
  int v30; // eax
  __int64 v31; // rdx
  __int64 v32; // r8
  unsigned int v33; // ebx
  unsigned __int8 *v34; // r12
  __int64 v36; // rdx
  unsigned int v37; // edx
  __int64 v38; // rdx
  __int128 v39; // [rsp-20h] [rbp-C0h]
  __int128 v40; // [rsp-20h] [rbp-C0h]
  __int64 v41; // [rsp+8h] [rbp-98h]
  __int64 v42; // [rsp+8h] [rbp-98h]
  _QWORD *v43; // [rsp+10h] [rbp-90h]
  __int16 v44; // [rsp+22h] [rbp-7Eh]
  unsigned int v45; // [rsp+28h] [rbp-78h]
  unsigned int v46; // [rsp+28h] [rbp-78h]
  __int16 v47; // [rsp+2Ah] [rbp-76h]
  __int64 v48; // [rsp+30h] [rbp-70h] BYREF
  int v49; // [rsp+38h] [rbp-68h]
  unsigned __int16 v50; // [rsp+40h] [rbp-60h] BYREF
  __int64 v51; // [rsp+48h] [rbp-58h]
  __int16 v52; // [rsp+50h] [rbp-50h] BYREF
  __int64 v53; // [rsp+58h] [rbp-48h]

  v4 = *(_QWORD *)(a2 + 80);
  v48 = v4;
  if ( v4 )
    sub_B96E90((__int64)&v48, v4, 1);
  v5 = *a1;
  v49 = *(_DWORD *)(a2 + 72);
  v6 = *(unsigned __int64 **)(a2 + 40);
  v7 = *v6;
  v8 = v6[1];
  v9 = *(_QWORD *)(*v6 + 48) + 16LL * *((unsigned int *)v6 + 2);
  v10 = *(_WORD *)v9;
  v11 = *(_QWORD *)(v9 + 8);
  v12 = a1[1];
  v50 = v10;
  v13 = *(_QWORD *)(v12 + 64);
  v51 = v11;
  sub_2FE6CC0((__int64)&v52, v5, v13, v10, v11);
  if ( (_BYTE)v52 == 5 )
  {
    v22 = sub_37946F0((__int64)a1, v7, v8);
    v26 = v37;
  }
  else
  {
    if ( v50 )
    {
      v17 = 0;
      LOWORD(v18) = word_4456580[v50 - 1];
    }
    else
    {
      v18 = sub_3009970((__int64)&v50, v5, v14, v15, v16);
      v47 = HIWORD(v18);
      v17 = v36;
    }
    HIWORD(v19) = v47;
    v41 = v17;
    LOWORD(v19) = v18;
    v43 = (_QWORD *)a1[1];
    v45 = v19;
    *(_QWORD *)&v20 = sub_3400EE0((__int64)v43, 0, (__int64)&v48, 0, a3);
    *((_QWORD *)&v39 + 1) = v8;
    *(_QWORD *)&v39 = v7;
    v22 = (__int64)sub_3406EB0(v43, 0x9Eu, (__int64)&v48, v45, v41, v21, v39, v20);
    v26 = v25;
  }
  v27 = *(unsigned __int16 **)(a2 + 48);
  v28 = (_QWORD *)a1[1];
  v29 = *(_QWORD *)(a2 + 40);
  v30 = *v27;
  v31 = *((_QWORD *)v27 + 1);
  v52 = v30;
  v53 = v31;
  if ( (_WORD)v30 )
  {
    v32 = 0;
    LOWORD(v30) = word_4456580[v30 - 1];
  }
  else
  {
    v42 = v29;
    v46 = v26;
    v30 = sub_3009970((__int64)&v52, v22, v31, v26, v23);
    v29 = v42;
    v44 = HIWORD(v30);
    v26 = v46;
    v32 = v38;
  }
  HIWORD(v33) = v44;
  LOWORD(v33) = v30;
  *((_QWORD *)&v40 + 1) = v26 | v8 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v40 = v22;
  v34 = sub_3406EB0(v28, 0xE6u, (__int64)&v48, v33, v32, v24, v40, *(_OWORD *)(v29 + 40));
  if ( v48 )
    sub_B91220((__int64)&v48, v48);
  return v34;
}
