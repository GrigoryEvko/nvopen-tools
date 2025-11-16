// Function: sub_114A350
// Address: 0x114a350
//
void __fastcall sub_114A350(_QWORD *a1, unsigned int a2, char a3)
{
  _BYTE *v5; // rax
  __int64 v6; // rdi
  _BYTE *v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rsi
  unsigned int **v10; // rdi
  __int64 v11; // rax
  unsigned int **v12; // rdi
  __int64 v13; // rax
  __int64 v14; // rsi
  __int64 v15; // rbx
  __int64 *v16; // r13
  _QWORD *v17; // rax
  __int64 v18; // r9
  __int64 v19; // r15
  __int64 v20; // rbx
  __int64 v21; // r13
  __int64 v22; // rdx
  unsigned int v23; // esi
  __int64 v24; // [rsp-8h] [rbp-98h]
  __int64 v25; // [rsp+8h] [rbp-88h]
  char v26; // [rsp+14h] [rbp-7Ch]
  unsigned int v27; // [rsp+1Ch] [rbp-74h] BYREF
  _BYTE *v28[2]; // [rsp+20h] [rbp-70h] BYREF
  __int64 v29; // [rsp+30h] [rbp-60h] BYREF
  __int64 v30; // [rsp+38h] [rbp-58h]
  __int16 v31; // [rsp+50h] [rbp-40h]

  v5 = (_BYTE *)a1[15];
  v6 = a1[14];
  v27 = a2;
  v28[0] = v5;
  v7 = (_BYTE *)sub_ACD640(v6, a2, 0);
  v8 = a1[2];
  v28[1] = v7;
  v9 = a1[1];
  v10 = *(unsigned int ***)(*a1 + 32LL);
  v11 = a1[3];
  v31 = 261;
  v29 = v11;
  v30 = a1[4];
  v25 = sub_921130(v10, v9, v8, v28, 2, (__int64)&v29, 3u);
  v12 = *(unsigned int ***)(*a1 + 32LL);
  v13 = a1[9];
  v31 = 261;
  v14 = a1[8];
  v29 = v13;
  v30 = a1[10];
  v15 = sub_94D3D0(v12, v14, (__int64)&v27, 1, (__int64)&v29);
  v26 = a3;
  v16 = *(__int64 **)(*a1 + 32LL);
  v31 = 257;
  v17 = sub_BD2C40(80, unk_3F10A10);
  v18 = v24;
  v19 = (__int64)v17;
  if ( v17 )
    sub_B4D3C0((__int64)v17, v15, v25, 0, v26, v25, 0, 0);
  (*(void (__fastcall **)(__int64, __int64, __int64 *, __int64, __int64, __int64))(*(_QWORD *)v16[11] + 16LL))(
    v16[11],
    v19,
    &v29,
    v16[7],
    v16[8],
    v18);
  v20 = *v16;
  v21 = *v16 + 16LL * *((unsigned int *)v16 + 2);
  while ( v21 != v20 )
  {
    v22 = *(_QWORD *)(v20 + 8);
    v23 = *(_DWORD *)v20;
    v20 += 16;
    sub_B99FD0(v19, v23, v22);
  }
  sub_B91FC0(&v29, a1[16]);
  sub_B9A100(v19, &v29);
}
