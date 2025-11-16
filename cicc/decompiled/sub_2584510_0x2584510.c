// Function: sub_2584510
// Address: 0x2584510
//
__int64 __fastcall sub_2584510(signed int **a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  signed int *v7; // rax
  int v8; // edi
  signed int v9; // r14d
  void *v10; // rax
  __int64 v11; // rdx
  __int64 v13; // rax
  __int64 v14; // rdi
  __int64 (__fastcall *v15)(__int64); // rax
  __int64 v16; // r14
  __int64 v17; // r12
  unsigned __int8 *v18; // rdi
  __int64 (*v19)(void); // rax
  char v20; // cl
  char v21; // al
  __int64 v22; // [rsp+0h] [rbp-100h] BYREF
  __int64 v23; // [rsp+8h] [rbp-F8h]
  void *v24; // [rsp+10h] [rbp-F0h] BYREF
  __int64 *v25; // [rsp+18h] [rbp-E8h] BYREF
  __int64 v26; // [rsp+20h] [rbp-E0h]
  __int64 v27; // [rsp+28h] [rbp-D8h] BYREF
  __int64 v28; // [rsp+30h] [rbp-D0h]
  __int64 v29; // [rsp+38h] [rbp-C8h]
  __int64 v30; // [rsp+40h] [rbp-C0h]
  char *v31; // [rsp+48h] [rbp-B8h] BYREF
  __int64 v32; // [rsp+50h] [rbp-B0h]
  char v33; // [rsp+58h] [rbp-A8h] BYREF
  char v34; // [rsp+D8h] [rbp-28h]

  v7 = *a1;
  v8 = *(_DWORD *)(a2 + 16);
  v9 = *v7;
  v10 = *(void **)a2;
  v25 = &v27;
  v26 = 0;
  v24 = v10;
  if ( v8 )
    sub_2538550((__int64)&v25, a2 + 8, a3, a4, a5, a6);
  v22 = sub_254CA10((__int64)&v24, v9);
  v23 = v11;
  if ( v25 != &v27 )
    _libc_free((unsigned __int64)v25);
  if ( !(unsigned __int8)sub_2509800(&v22) )
    return 0;
  v13 = sub_25803A0((__int64)a1[1], v22, v23, (__int64)a1[2], 0, 0, 1);
  v14 = v13;
  if ( !v13 )
    return 0;
  v15 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v13 + 48LL);
  if ( v15 == sub_2534B10 )
    v16 = v14 + 88;
  else
    v16 = v15(v14);
  v17 = (__int64)a1[3];
  if ( !*(_BYTE *)(v17 + 208) )
  {
    v34 = 0;
    v29 = 0;
    v27 = 0;
    LOWORD(v26) = 256;
    v24 = &unk_4A170B8;
    v28 = 0;
    v31 = &v33;
    v25 = (__int64 *)&unk_4A16CD8;
    v30 = 0;
    v32 = 0x800000000LL;
    *(_QWORD *)v17 = &unk_4A170B8;
    *(_BYTE *)(v17 + 16) = v26;
    v20 = BYTE1(v26);
    *(_QWORD *)(v17 + 40) = 0;
    *(_QWORD *)(v17 + 32) = 0;
    *(_DWORD *)(v17 + 48) = 0;
    *(_BYTE *)(v17 + 17) = v20;
    *(_QWORD *)(v17 + 8) = &unk_4A16CD8;
    *(_QWORD *)(v17 + 24) = 1;
    ++v27;
    *(_QWORD *)(v17 + 32) = v28;
    *(_QWORD *)(v17 + 40) = v29;
    *(_DWORD *)(v17 + 48) = v30;
    v28 = 0;
    v29 = 0;
    LODWORD(v30) = 0;
    *(_QWORD *)(v17 + 56) = v17 + 72;
    *(_QWORD *)(v17 + 64) = 0x800000000LL;
    if ( (_DWORD)v32 )
      sub_2560A10((unsigned int *)(v17 + 56), (__int64)&v31);
    v21 = v34;
    *(_BYTE *)(v17 + 208) = 1;
    *(_BYTE *)(v17 + 200) = v21;
    sub_25485A0((__int64)&v24);
    v17 = (__int64)a1[3];
  }
  *(_WORD *)(v17 + 16) &= *(_WORD *)(v16 + 16);
  sub_2576560(v17, v16);
  sub_2560F70((__int64)&v24, v17);
  sub_25485A0((__int64)&v24);
  v18 = (unsigned __int8 *)a1[3];
  v19 = *(__int64 (**)(void))(*(_QWORD *)v18 + 16LL);
  if ( (char *)v19 == (char *)sub_2505E40 )
    return v18[17];
  else
    return v19();
}
