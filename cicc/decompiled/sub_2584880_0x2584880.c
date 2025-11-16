// Function: sub_2584880
// Address: 0x2584880
//
__int64 __fastcall sub_2584880(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rdi
  __int64 (__fastcall *v7)(__int64); // rax
  __int64 v8; // r13
  __int64 v9; // r12
  unsigned __int8 *v10; // rdi
  __int64 (*v11)(void); // rax
  char v13; // cl
  char v14; // al
  _QWORD v15[2]; // [rsp+0h] [rbp-F0h] BYREF
  __int16 v16; // [rsp+10h] [rbp-E0h]
  __int64 v17; // [rsp+18h] [rbp-D8h]
  __int64 v18; // [rsp+20h] [rbp-D0h]
  __int64 v19; // [rsp+28h] [rbp-C8h]
  __int64 v20; // [rsp+30h] [rbp-C0h]
  char *v21; // [rsp+38h] [rbp-B8h] BYREF
  __int64 v22; // [rsp+40h] [rbp-B0h]
  char v23; // [rsp+48h] [rbp-A8h] BYREF
  char v24; // [rsp+C8h] [rbp-28h]

  v3 = sub_250D2C0(a2, **(_QWORD **)a1);
  v5 = sub_25803A0(*(_QWORD *)(a1 + 8), v3, v4, *(_QWORD *)(a1 + 16), 0, 0, 1);
  if ( !v5 )
    return 0;
  v6 = v5;
  v7 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v5 + 48LL);
  if ( v7 == sub_2534B10 )
    v8 = v6 + 88;
  else
    v8 = v7(v6);
  v9 = *(_QWORD *)(a1 + 24);
  if ( !*(_BYTE *)(v9 + 208) )
  {
    v24 = 0;
    v19 = 0;
    v17 = 0;
    v16 = 256;
    v15[0] = &unk_4A170B8;
    v18 = 0;
    v21 = &v23;
    v15[1] = &unk_4A16CD8;
    v20 = 0;
    v22 = 0x800000000LL;
    *(_QWORD *)v9 = &unk_4A170B8;
    *(_BYTE *)(v9 + 16) = v16;
    v13 = HIBYTE(v16);
    *(_QWORD *)(v9 + 40) = 0;
    *(_QWORD *)(v9 + 32) = 0;
    *(_DWORD *)(v9 + 48) = 0;
    *(_BYTE *)(v9 + 17) = v13;
    *(_QWORD *)(v9 + 8) = &unk_4A16CD8;
    *(_QWORD *)(v9 + 24) = 1;
    ++v17;
    *(_QWORD *)(v9 + 32) = v18;
    *(_QWORD *)(v9 + 40) = v19;
    *(_DWORD *)(v9 + 48) = v20;
    v18 = 0;
    v19 = 0;
    LODWORD(v20) = 0;
    *(_QWORD *)(v9 + 56) = v9 + 72;
    *(_QWORD *)(v9 + 64) = 0x800000000LL;
    if ( (_DWORD)v22 )
      sub_2560A10((unsigned int *)(v9 + 56), (__int64)&v21);
    v14 = v24;
    *(_BYTE *)(v9 + 208) = 1;
    *(_BYTE *)(v9 + 200) = v14;
    sub_25485A0((__int64)v15);
    v9 = *(_QWORD *)(a1 + 24);
  }
  *(_WORD *)(v9 + 16) &= *(_WORD *)(v8 + 16);
  sub_2576560(v9, v8);
  sub_2560F70((__int64)v15, v9);
  sub_25485A0((__int64)v15);
  v10 = *(unsigned __int8 **)(a1 + 24);
  v11 = *(__int64 (**)(void))(*(_QWORD *)v10 + 16LL);
  if ( (char *)v11 == (char *)sub_2505E40 )
    return v10[17];
  else
    return v11();
}
