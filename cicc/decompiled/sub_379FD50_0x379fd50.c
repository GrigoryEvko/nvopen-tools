// Function: sub_379FD50
// Address: 0x379fd50
//
unsigned __int8 *__fastcall sub_379FD50(__int64 *a1, __int64 a2, __m128i a3)
{
  __int64 (__fastcall *v5)(__int64, __int64, unsigned int, __int64); // rbx
  __int16 *v6; // rax
  unsigned __int16 v7; // si
  __int64 v8; // r8
  __int64 v9; // rax
  __int64 *v10; // rax
  __int64 v11; // rsi
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // r10
  __int64 v15; // rcx
  unsigned __int8 *v16; // r12
  unsigned __int64 v17; // r13
  unsigned __int16 *v18; // rax
  int v19; // edx
  __int64 v20; // rax
  __int64 v21; // rsi
  _QWORD *v22; // r9
  unsigned int v23; // esi
  unsigned __int8 *v24; // r12
  unsigned __int16 v26; // di
  unsigned __int16 v27; // ax
  __int64 v28; // rdx
  unsigned int v29; // esi
  int v30; // esi
  int v31; // eax
  __int64 v32; // r10
  __int64 v33; // rcx
  __int64 v34; // r8
  __int64 v35; // rdx
  __int64 v36; // r8
  unsigned __int16 v37; // ax
  __int64 v38; // rdx
  unsigned int v39; // edi
  unsigned int v40; // edx
  __int64 v41; // rdx
  __int128 v42; // [rsp-10h] [rbp-C0h]
  __int16 v43; // [rsp+2h] [rbp-AEh]
  __int64 v44; // [rsp+8h] [rbp-A8h]
  __int64 v45; // [rsp+10h] [rbp-A0h]
  __int64 v46; // [rsp+10h] [rbp-A0h]
  __int64 v47; // [rsp+10h] [rbp-A0h]
  _QWORD *v48; // [rsp+18h] [rbp-98h]
  __int64 v49; // [rsp+18h] [rbp-98h]
  __int64 v50; // [rsp+18h] [rbp-98h]
  __int128 v51; // [rsp+20h] [rbp-90h]
  unsigned int v52; // [rsp+40h] [rbp-70h] BYREF
  __int64 v53; // [rsp+48h] [rbp-68h]
  __int16 v54; // [rsp+50h] [rbp-60h] BYREF
  __int64 v55; // [rsp+58h] [rbp-58h]
  __int64 v56; // [rsp+60h] [rbp-50h] BYREF
  int v57; // [rsp+68h] [rbp-48h]
  __int64 v58; // [rsp+70h] [rbp-40h]

  v5 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  v6 = *(__int16 **)(a2 + 48);
  v7 = *v6;
  v8 = *((_QWORD *)v6 + 1);
  v9 = a1[1];
  if ( v5 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v56, *a1, *(_QWORD *)(v9 + 64), v7, v8);
    LOWORD(v52) = v57;
    v53 = v58;
  }
  else
  {
    v52 = v5(*a1, *(_QWORD *)(v9 + 64), v7, v8);
    v53 = v41;
  }
  v10 = *(__int64 **)(a2 + 40);
  v11 = *v10;
  *(_QWORD *)&v51 = sub_379AB60((__int64)a1, *v10, v10[1]);
  v12 = *(_QWORD *)(a2 + 40);
  *((_QWORD *)&v51 + 1) = v13;
  v14 = *(_QWORD *)(v12 + 40);
  v15 = *(_QWORD *)(v12 + 48);
  v16 = (unsigned __int8 *)v14;
  v17 = v15;
  v18 = (unsigned __int16 *)(*(_QWORD *)(v14 + 48) + 16LL * *(unsigned int *)(v12 + 48));
  v19 = *v18;
  v20 = *((_QWORD *)v18 + 1);
  v54 = v19;
  v55 = v20;
  if ( (_WORD)v19 )
  {
    if ( (unsigned __int16)(v19 - 17) > 0xD3u )
      goto LABEL_5;
    v26 = word_4456580[v19 - 1];
    v27 = v52;
    v28 = 0;
    v29 = v26;
    if ( (_WORD)v52 )
    {
LABEL_11:
      v45 = v15;
      v49 = v14;
      v30 = word_4456340[v27 - 1];
      if ( (unsigned __int16)(v27 - 176) > 0x34u )
      {
        LOWORD(v31) = sub_2D43050(v26, v30);
        v33 = v45;
        v32 = v49;
      }
      else
      {
        LOWORD(v31) = sub_2D43AD0(v26, v30);
        v32 = v49;
        v33 = v45;
      }
      v34 = 0;
      goto LABEL_17;
    }
  }
  else
  {
    v44 = v15;
    v46 = v14;
    if ( !sub_30070B0((__int64)&v54) )
      goto LABEL_5;
    v37 = sub_3009970((__int64)&v54, v11, v35, v44, v36);
    v15 = v44;
    v14 = v46;
    v26 = v37;
    v27 = v52;
    v29 = v26;
    if ( (_WORD)v52 )
      goto LABEL_11;
  }
  v47 = v15;
  v50 = v14;
  v31 = sub_3009490((unsigned __int16 *)&v52, v29, v28);
  v33 = v47;
  v32 = v50;
  v43 = HIWORD(v31);
  v34 = v38;
LABEL_17:
  HIWORD(v39) = v43;
  LOWORD(v39) = v31;
  v16 = sub_3790540((__int64)a1, v32, v33, v39, v34, 0, a3);
  v17 = v40 | v17 & 0xFFFFFFFF00000000LL;
LABEL_5:
  v21 = *(_QWORD *)(a2 + 80);
  v22 = (_QWORD *)a1[1];
  v56 = v21;
  if ( v21 )
  {
    v48 = v22;
    sub_B96E90((__int64)&v56, v21, 1);
    v22 = v48;
  }
  *((_QWORD *)&v42 + 1) = v17;
  v23 = *(_DWORD *)(a2 + 24);
  *(_QWORD *)&v42 = v16;
  v57 = *(_DWORD *)(a2 + 72);
  v24 = sub_3406EB0(v22, v23, (__int64)&v56, v52, v53, (__int64)v22, v51, v42);
  if ( v56 )
    sub_B91220((__int64)&v56, v56);
  return v24;
}
