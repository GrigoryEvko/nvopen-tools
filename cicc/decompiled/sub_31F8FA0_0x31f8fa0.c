// Function: sub_31F8FA0
// Address: 0x31f8fa0
//
void __fastcall sub_31F8FA0(_QWORD *a1)
{
  __int64 v2; // rax
  __int64 v3; // rdi
  __int64 v4; // r13
  _BYTE *v5; // rax
  _BYTE *v6; // rcx
  _BYTE *v7; // rbx
  unsigned __int8 v8; // al
  _QWORD *v9; // rcx
  unsigned __int8 v10; // al
  unsigned __int64 v11; // rdi
  __int64 v12; // rdx
  __int64 v13; // rax
  unsigned __int8 v14; // al
  unsigned __int64 v15; // rdi
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  unsigned __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  unsigned __int64 v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rax
  int v28; // eax
  __m128i v29; // xmm0
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // r13
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // r8
  __int64 v38; // rax
  __int64 v39; // rdi
  __int64 v40; // r14
  void (*v41)(); // rax
  _BYTE *v42; // [rsp+8h] [rbp-C8h]
  unsigned int v43; // [rsp+1Ch] [rbp-B4h]
  __m128i v44; // [rsp+20h] [rbp-B0h] BYREF
  int v45; // [rsp+30h] [rbp-A0h]
  const char *v46; // [rsp+40h] [rbp-90h] BYREF
  char v47; // [rsp+60h] [rbp-70h]
  char v48; // [rsp+61h] [rbp-6Fh]
  __int16 v49; // [rsp+70h] [rbp-60h] BYREF
  int v50; // [rsp+72h] [rbp-5Eh]
  const char *v51; // [rsp+78h] [rbp-58h]
  __int64 v52; // [rsp+80h] [rbp-50h]
  __m128i v53; // [rsp+88h] [rbp-48h] BYREF
  int v54; // [rsp+98h] [rbp-38h]

  v2 = a1[2];
  v44 = 0u;
  v3 = *(_QWORD *)(v2 + 2488);
  v45 = 0;
  v4 = sub_BA8DC0(v3, (__int64)"llvm.dbg.cu", 11);
  sub_B91A00(v4);
  v5 = (_BYTE *)sub_B91A10(v4, 0);
  v6 = v5;
  v7 = v5;
  if ( *v5 != 16 )
  {
    v8 = *(v5 - 16);
    if ( (v8 & 2) != 0 )
      v9 = (_QWORD *)*((_QWORD *)v6 - 4);
    else
      v9 = &v6[-8 * ((v8 >> 2) & 0xF) - 16];
    v7 = (_BYTE *)*v9;
  }
  v42 = v7 - 16;
  v10 = *(v7 - 16);
  if ( (v10 & 2) != 0 )
  {
    v11 = *(_QWORD *)(*((_QWORD *)v7 - 4) + 8LL);
    if ( v11 )
    {
LABEL_7:
      v11 = sub_B91420(v11);
      goto LABEL_8;
    }
  }
  else
  {
    v11 = *(_QWORD *)&v42[-8 * ((v10 >> 2) & 0xF) + 8];
    if ( v11 )
      goto LABEL_7;
  }
  v12 = 0;
LABEL_8:
  v51 = (const char *)v11;
  v49 = 5637;
  LODWORD(v46) = 0;
  v50 = 0;
  v52 = v12;
  v13 = sub_370B390(a1 + 81, &v49);
  v44.m128i_i32[0] = sub_3707F80(a1 + 79, v13);
  v14 = *(v7 - 16);
  if ( (v14 & 2) != 0 )
  {
    v15 = **((_QWORD **)v7 - 4);
    if ( v15 )
    {
LABEL_10:
      v15 = sub_B91420(v15);
      goto LABEL_11;
    }
  }
  else
  {
    v15 = *(_QWORD *)&v42[-8 * ((v14 >> 2) & 0xF)];
    if ( v15 )
      goto LABEL_10;
  }
  v16 = 0;
LABEL_11:
  v51 = (const char *)v15;
  v49 = 5637;
  v50 = 0;
  v52 = v16;
  v17 = sub_370B390(a1 + 81, &v49);
  v44.m128i_i32[2] = sub_3707F80(a1 + 79, v17);
  LODWORD(v46) = 0;
  v49 = 5637;
  v50 = 0;
  v51 = byte_3F871B3;
  v52 = 0;
  v18 = sub_370B390(a1 + 81, &v49);
  v44.m128i_i32[3] = sub_3707F80(a1 + 79, v18);
  v19 = a1[1];
  LODWORD(v46) = 0;
  v20 = *(_QWORD *)(v19 + 200);
  v21 = *(_QWORD *)(v20 + 1136);
  v22 = *(_QWORD *)(v20 + 1144);
  v49 = 5637;
  v50 = 0;
  v51 = (const char *)v21;
  v52 = v22;
  v23 = sub_370B390(a1 + 81, &v49);
  LODWORD(v46) = 0;
  v44.m128i_i32[1] = sub_3707F80(a1 + 79, v23);
  v24 = *(_QWORD *)(a1[1] + 200LL);
  v25 = *(_QWORD *)(v24 + 1168);
  v26 = *(_QWORD *)(v24 + 1176);
  v49 = 5637;
  v50 = 0;
  v51 = (const char *)v25;
  v52 = v26;
  v27 = sub_370B390(a1 + 81, &v49);
  v28 = sub_3707F80(a1 + 79, v27);
  v29 = _mm_loadu_si128(&v44);
  v51 = (const char *)&v53;
  v45 = v28;
  v49 = 5635;
  v54 = v28;
  v53 = v29;
  v52 = 0x500000005LL;
  v30 = sub_370B100(a1 + 81, &v49);
  v43 = sub_3707F80(a1 + 79, v30);
  v34 = sub_31F8650((__int64)a1, 241, v31, v32, v33);
  v38 = sub_31F8790((__int64)a1, 4428, v35, v36, v37);
  v39 = a1[66];
  v40 = v38;
  v41 = *(void (**)())(*(_QWORD *)v39 + 120LL);
  v48 = 1;
  v46 = "LF_BUILDINFO index";
  v47 = 3;
  if ( v41 != nullsub_98 )
  {
    ((void (__fastcall *)(__int64, const char **, __int64))v41)(v39, &v46, 1);
    v39 = a1[66];
  }
  (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v39 + 536LL))(v39, v43, 4);
  sub_31F8930((__int64)a1, v40);
  sub_31F8740((__int64)a1, v34);
  if ( v51 != (const char *)&v53 )
    _libc_free((unsigned __int64)v51);
}
