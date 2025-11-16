// Function: sub_381E9D0
// Address: 0x381e9d0
//
void __fastcall sub_381E9D0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __m128i a5)
{
  __int64 v6; // r9
  __int64 (__fastcall *v7)(__int64, __int64, unsigned int, __int64); // r10
  __int16 *v8; // rax
  unsigned __int16 v9; // si
  __int64 v10; // r8
  __int64 v11; // rax
  unsigned int v12; // eax
  __int64 v13; // rax
  __int64 v14; // rdx
  unsigned int v15; // eax
  __int64 v16; // r10
  __int64 v17; // rsi
  unsigned int v18; // r12d
  bool v19; // r8
  bool v20; // cl
  int v21; // eax
  __int64 v22; // r11
  unsigned __int8 v23; // r14
  unsigned __int8 *v24; // rax
  bool v25; // cc
  __int64 v26; // r10
  int v27; // edx
  unsigned int v28; // eax
  __int64 v29; // rbx
  unsigned __int8 *v30; // rax
  int v31; // edx
  __int64 v32; // rdx
  __int64 v33; // [rsp+0h] [rbp-E0h]
  __int64 v34; // [rsp+0h] [rbp-E0h]
  bool v35; // [rsp+8h] [rbp-D8h]
  __int64 v36; // [rsp+8h] [rbp-D8h]
  char **v39; // [rsp+20h] [rbp-C0h]
  bool v40; // [rsp+2Ch] [rbp-B4h]
  unsigned __int8 v41; // [rsp+2Ch] [rbp-B4h]
  unsigned int v42; // [rsp+50h] [rbp-90h] BYREF
  __int64 v43; // [rsp+58h] [rbp-88h]
  __int64 v44; // [rsp+60h] [rbp-80h] BYREF
  int v45; // [rsp+68h] [rbp-78h]
  unsigned __int64 v46; // [rsp+70h] [rbp-70h] BYREF
  unsigned int v47; // [rsp+78h] [rbp-68h]
  unsigned __int64 v48; // [rsp+80h] [rbp-60h] BYREF
  unsigned int v49; // [rsp+88h] [rbp-58h]
  __int64 v50; // [rsp+90h] [rbp-50h] BYREF
  __int64 v51; // [rsp+98h] [rbp-48h]
  __int64 v52; // [rsp+A0h] [rbp-40h]

  v6 = *a1;
  v7 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  v8 = *(__int16 **)(a2 + 48);
  v9 = *v8;
  v10 = *((_QWORD *)v8 + 1);
  v11 = a1[1];
  if ( v7 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v50, v6, *(_QWORD *)(v11 + 64), v9, v10);
    LOWORD(v12) = v51;
    LOWORD(v42) = v51;
    v43 = v52;
  }
  else
  {
    v12 = v7(v6, *(_QWORD *)(v11 + 64), v9, v10);
    v42 = v12;
    v43 = v32;
  }
  if ( (_WORD)v12 )
  {
    if ( (_WORD)v12 == 1 || (unsigned __int16)(v12 - 504) <= 7u )
      BUG();
    v14 = 16LL * ((unsigned __int16)v12 - 1);
    v13 = *(_QWORD *)&byte_444C4A0[v14];
    LOBYTE(v14) = byte_444C4A0[v14 + 8];
  }
  else
  {
    v13 = sub_3007260((__int64)&v42);
    v50 = v13;
    v51 = v14;
  }
  v48 = v13;
  LOBYTE(v49) = v14;
  v15 = sub_CA1930(&v48);
  v16 = *(_QWORD *)(a2 + 96);
  v17 = *(_QWORD *)(a2 + 80);
  v18 = v15;
  v19 = *(_DWORD *)(a2 + 24) > 499;
  v39 = (char **)(v16 + 24);
  v20 = (*(_BYTE *)(a2 + 32) & 8) != 0;
  v44 = v17;
  if ( v17 )
  {
    v33 = v16;
    v35 = v19;
    v40 = v20;
    sub_B96E90((__int64)&v44, v17, 1);
    v16 = v33;
    v19 = v35;
    v20 = v40;
  }
  v21 = *(_DWORD *)(a2 + 72);
  v22 = a1[1];
  v34 = v16;
  v23 = v19;
  v45 = v21;
  v36 = v22;
  v41 = v20;
  sub_C44740((__int64)&v48, v39, v18);
  v24 = sub_34007B0(v36, (__int64)&v48, (__int64)&v44, v42, v43, v23, a5, v41);
  v25 = v49 <= 0x40;
  v26 = v34;
  *(_QWORD *)a3 = v24;
  *(_DWORD *)(a3 + 8) = v27;
  if ( !v25 && v48 )
  {
    j_j___libc_free_0_0(v48);
    v26 = v34;
  }
  v28 = *(_DWORD *)(v26 + 32);
  v29 = a1[1];
  v47 = v28;
  if ( v28 > 0x40 )
  {
    sub_C43780((__int64)&v46, (const void **)v39);
    v28 = v47;
    if ( v47 > 0x40 )
    {
      sub_C482E0((__int64)&v46, v18);
      goto LABEL_14;
    }
  }
  else
  {
    v46 = *(_QWORD *)(v26 + 24);
  }
  if ( v18 == v28 )
    v46 = 0;
  else
    v46 >>= v18;
LABEL_14:
  sub_C44740((__int64)&v48, (char **)&v46, v18);
  v30 = sub_34007B0(v29, (__int64)&v48, (__int64)&v44, v42, v43, v23, a5, v41);
  v25 = v49 <= 0x40;
  *(_QWORD *)a4 = v30;
  *(_DWORD *)(a4 + 8) = v31;
  if ( !v25 && v48 )
    j_j___libc_free_0_0(v48);
  if ( v47 > 0x40 && v46 )
    j_j___libc_free_0_0(v46);
  if ( v44 )
    sub_B91220((__int64)&v44, v44);
}
