// Function: sub_382E910
// Address: 0x382e910
//
__m128i *__fastcall sub_382E910(__int64 *a1, __int64 a2, __m128i a3)
{
  __int16 *v4; // rax
  __int64 v5; // r11
  __int16 v6; // dx
  unsigned __int64 *v7; // rax
  __int64 v8; // rax
  unsigned __int16 v9; // r14
  __int64 v10; // rbx
  __int64 (__fastcall *v11)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v12; // rsi
  __int64 v13; // rsi
  __m128i *v14; // r12
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // rdx
  __int64 v20; // r14
  __int64 v21; // rdx
  __int64 v22; // rdx
  __int64 *v23; // rbx
  int v24; // eax
  __int64 v25; // r8
  unsigned int v26; // ebx
  int v27; // r9d
  __int128 v28; // rax
  _QWORD *v29; // r13
  __int128 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rdx
  __int16 v33; // [rsp+2h] [rbp-DEh]
  __int64 v34; // [rsp+8h] [rbp-D8h]
  unsigned int v35; // [rsp+10h] [rbp-D0h]
  __int64 v36; // [rsp+10h] [rbp-D0h]
  unsigned __int64 v37; // [rsp+18h] [rbp-C8h]
  unsigned __int64 v38; // [rsp+20h] [rbp-C0h]
  __int128 v39; // [rsp+20h] [rbp-C0h]
  unsigned int v40; // [rsp+30h] [rbp-B0h] BYREF
  unsigned __int64 v41; // [rsp+38h] [rbp-A8h]
  unsigned __int16 v42[4]; // [rsp+40h] [rbp-A0h] BYREF
  __int64 v43; // [rsp+48h] [rbp-98h]
  __int64 v44; // [rsp+50h] [rbp-90h] BYREF
  int v45; // [rsp+58h] [rbp-88h]
  _QWORD v46[2]; // [rsp+60h] [rbp-80h] BYREF
  unsigned __int64 v47; // [rsp+70h] [rbp-70h]
  __int64 v48; // [rsp+78h] [rbp-68h]
  unsigned __int64 v49; // [rsp+80h] [rbp-60h]
  __int64 v50; // [rsp+88h] [rbp-58h]
  _BYTE v51[8]; // [rsp+90h] [rbp-50h] BYREF
  unsigned __int16 v52; // [rsp+98h] [rbp-48h]
  __int64 v53; // [rsp+A0h] [rbp-40h]

  v4 = *(__int16 **)(a2 + 48);
  v5 = *a1;
  v6 = *v4;
  v41 = *((_QWORD *)v4 + 1);
  v7 = *(unsigned __int64 **)(a2 + 40);
  LOWORD(v40) = v6;
  v38 = *v7;
  v37 = v7[1];
  v8 = *(_QWORD *)(*v7 + 48) + 16LL * *((unsigned int *)v7 + 2);
  v9 = *(_WORD *)v8;
  v10 = *(_QWORD *)(v8 + 8);
  v11 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v5 + 592LL);
  if ( v11 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)v51, v5, *(_QWORD *)(a1[1] + 64), v9, v10);
    v42[0] = v52;
    v43 = v53;
  }
  else
  {
    *(_DWORD *)v42 = v11(v5, *(_QWORD *)(a1[1] + 64), v9, v10);
    v43 = v31;
  }
  v12 = *(_QWORD *)(a2 + 80);
  v44 = v12;
  if ( v12 )
    sub_B96E90((__int64)&v44, v12, 1);
  v13 = *a1;
  v45 = *(_DWORD *)(a2 + 72);
  sub_2FE6CC0((__int64)v51, v13, *(_QWORD *)(a1[1] + 64), v9, v10);
  if ( v51[0] != 1 )
    goto LABEL_8;
  if ( (_WORD)v40 )
  {
    if ( (unsigned __int16)(v40 - 17) > 0xD3u )
    {
LABEL_8:
      v14 = sub_375AC00((__int64)a1, v38, v37, v40, v41);
      goto LABEL_9;
    }
  }
  else if ( !sub_30070B0((__int64)&v40) )
  {
    goto LABEL_8;
  }
  if ( *(_BYTE *)sub_2E79000(*(__int64 **)(a1[1] + 40)) )
    goto LABEL_8;
  LODWORD(v46[0]) = sub_3281170(&v40, v13, v16, v17, v18);
  v20 = v19;
  v46[1] = v19;
  v47 = sub_2D5B750((unsigned __int16 *)v46);
  v48 = v21;
  v49 = sub_2D5B750(v42);
  v50 = v22;
  if ( (_BYTE)v22 != (_BYTE)v48 || v49 % v47 )
    goto LABEL_8;
  v34 = v49 / v47;
  v35 = v46[0];
  v23 = *(__int64 **)(a1[1] + 64);
  LOWORD(v24) = sub_2D43050(v46[0], v49 / v47);
  v25 = 0;
  if ( !(_WORD)v24 )
  {
    v24 = sub_3009400(v23, v35, v20, v34, 0);
    v33 = HIWORD(v24);
    v25 = v32;
  }
  HIWORD(v26) = v33;
  v36 = v25;
  LOWORD(v26) = v24;
  sub_2FE6CC0((__int64)v51, *a1, *(_QWORD *)(a1[1] + 64), (unsigned __int16)v24, v25);
  if ( v51[0] )
    goto LABEL_8;
  sub_37AE0F0((__int64)a1, v38, v37);
  *(_QWORD *)&v28 = sub_33FAF80(a1[1], 234, (__int64)&v44, v26, v36, v27, a3);
  v29 = (_QWORD *)a1[1];
  v39 = v28;
  *(_QWORD *)&v30 = sub_3400EE0((__int64)v29, 0, (__int64)&v44, 0, a3);
  v14 = (__m128i *)sub_3406EB0(v29, 0xA1u, (__int64)&v44, v40, v41, *((__int64 *)&v39 + 1), v39, v30);
LABEL_9:
  if ( v44 )
    sub_B91220((__int64)&v44, v44);
  return v14;
}
