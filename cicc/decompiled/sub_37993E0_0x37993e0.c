// Function: sub_37993E0
// Address: 0x37993e0
//
unsigned __int8 *__fastcall sub_37993E0(_QWORD *a1, __int64 a2, __m128i a3)
{
  unsigned int v3; // r15d
  __int16 *v6; // rax
  __int16 v7; // dx
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rdx
  unsigned __int64 v11; // rsi
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r10
  unsigned int *v15; // rax
  __int64 v16; // rdx
  __int64 v17; // r11
  __int64 v18; // rdx
  unsigned __int16 *v19; // rax
  __int64 v20; // r9
  int v21; // eax
  __int64 v22; // rsi
  _QWORD *v23; // rdi
  int v24; // r9d
  _DWORD *v25; // rcx
  unsigned __int16 v26; // dx
  unsigned int v27; // eax
  bool v28; // al
  int v29; // r9d
  unsigned __int8 *v30; // r12
  __int64 v32; // rdx
  __int128 v33; // [rsp-20h] [rbp-D0h]
  __int64 v34; // [rsp+0h] [rbp-B0h]
  __int64 v35; // [rsp+0h] [rbp-B0h]
  __int64 v36; // [rsp+8h] [rbp-A8h]
  __int64 v37; // [rsp+8h] [rbp-A8h]
  unsigned int v38; // [rsp+1Ch] [rbp-94h]
  int v39; // [rsp+1Ch] [rbp-94h]
  char v40; // [rsp+1Ch] [rbp-94h]
  unsigned int v41; // [rsp+1Ch] [rbp-94h]
  __int128 v42; // [rsp+20h] [rbp-90h]
  _DWORD *v43; // [rsp+20h] [rbp-90h]
  __int64 v44; // [rsp+30h] [rbp-80h]
  __int64 v45; // [rsp+38h] [rbp-78h]
  unsigned int v46; // [rsp+50h] [rbp-60h] BYREF
  __int64 v47; // [rsp+58h] [rbp-58h]
  __int64 v48; // [rsp+60h] [rbp-50h] BYREF
  int v49; // [rsp+68h] [rbp-48h]
  __int16 v50; // [rsp+70h] [rbp-40h] BYREF
  __int64 v51; // [rsp+78h] [rbp-38h]

  v6 = *(__int16 **)(a2 + 48);
  v7 = *v6;
  v47 = *((_QWORD *)v6 + 1);
  v8 = *(_QWORD *)(a2 + 40);
  LOWORD(v46) = v7;
  *(_QWORD *)&v42 = sub_37946F0((__int64)a1, *(_QWORD *)v8, *(_QWORD *)(v8 + 8));
  v9 = *(_QWORD *)(a2 + 40);
  *((_QWORD *)&v42 + 1) = v10;
  v11 = *(_QWORD *)(v9 + 40);
  v14 = sub_37946F0((__int64)a1, v11, *(_QWORD *)(v9 + 48));
  v15 = *(unsigned int **)(a2 + 40);
  v17 = v16;
  v18 = *(_QWORD *)v15;
  v19 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)v15 + 48LL) + 16LL * v15[2]);
  v20 = *v19;
  v45 = *((_QWORD *)v19 + 1);
  if ( (_WORD)v46 )
  {
    v44 = 0;
    LOWORD(v21) = word_4456580[(unsigned __int16)v46 - 1];
  }
  else
  {
    v35 = v14;
    v37 = v17;
    v41 = *v19;
    v21 = sub_3009970((__int64)&v46, v11, v18, v12, v13);
    v14 = v35;
    v17 = v37;
    v44 = v32;
    v20 = v41;
    HIWORD(v3) = HIWORD(v21);
  }
  v22 = *(_QWORD *)(a2 + 80);
  LOWORD(v3) = v21;
  v48 = v22;
  if ( v22 )
  {
    v34 = v14;
    v36 = v17;
    v38 = v20;
    sub_B96E90((__int64)&v48, v22, 1);
    v14 = v34;
    v17 = v36;
    v20 = v38;
  }
  v23 = (_QWORD *)a1[1];
  v39 = v20;
  v49 = *(_DWORD *)(a2 + 72);
  *((_QWORD *)&v33 + 1) = v17;
  *(_QWORD *)&v33 = v14;
  sub_340F900(v23, 0xD0u, (__int64)&v48, 2u, 0, v20, v42, v33, *(_OWORD *)(*(_QWORD *)(a2 + 40) + 80LL));
  v24 = v39;
  v25 = (_DWORD *)*a1;
  v50 = v39;
  v51 = v45;
  if ( (_WORD)v39 )
  {
    v26 = v39 - 17;
    if ( (unsigned __int16)(v39 - 10) > 6u )
    {
      v24 = v39 - 126;
      if ( (unsigned __int16)(v39 - 126) > 0x31u )
      {
        if ( v26 <= 0xD3u )
        {
LABEL_9:
          v27 = v25[17];
          goto LABEL_13;
        }
LABEL_12:
        v27 = v25[15];
        goto LABEL_13;
      }
    }
    if ( v26 <= 0xD3u )
      goto LABEL_9;
  }
  else
  {
    v43 = v25;
    v40 = sub_3007030((__int64)&v50);
    v28 = sub_30070B0((__int64)&v50);
    v25 = v43;
    if ( v28 )
      goto LABEL_9;
    if ( !v40 )
      goto LABEL_12;
  }
  v27 = v25[16];
LABEL_13:
  if ( v27 > 2 )
    BUG();
  sub_33FAF80(a1[1], 215 - v27, (__int64)&v48, v3, v44, v24, a3);
  v30 = sub_33FAF80(a1[1], 167, (__int64)&v48, v46, v47, v29, a3);
  if ( v48 )
    sub_B91220((__int64)&v48, v48);
  return v30;
}
