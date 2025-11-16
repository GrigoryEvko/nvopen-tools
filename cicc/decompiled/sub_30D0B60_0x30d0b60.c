// Function: sub_30D0B60
// Address: 0x30d0b60
//
__int64 *__fastcall sub_30D0B60(__int64 *a1, __int64 a2, _QWORD *a3)
{
  __int64 v4; // r15
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // r15
  __int64 v8; // rax
  __int64 v9; // rbx
  bool v10; // zf
  __int64 v12; // rax
  unsigned int v13; // eax
  unsigned int v14; // eax
  char v15; // al
  __int64 v16; // [rsp+0h] [rbp-D0h] BYREF
  int v17; // [rsp+8h] [rbp-C8h]
  __int64 v18; // [rsp+10h] [rbp-C0h]
  unsigned __int64 v19; // [rsp+18h] [rbp-B8h] BYREF
  unsigned int v20; // [rsp+20h] [rbp-B0h]
  unsigned __int64 v21; // [rsp+28h] [rbp-A8h] BYREF
  unsigned int v22; // [rsp+30h] [rbp-A0h]
  char v23; // [rsp+38h] [rbp-98h]
  char v24; // [rsp+40h] [rbp-90h]
  __int64 v25; // [rsp+50h] [rbp-80h]
  int v26; // [rsp+58h] [rbp-78h]
  __int64 v27; // [rsp+60h] [rbp-70h]
  unsigned __int64 v28; // [rsp+68h] [rbp-68h] BYREF
  unsigned int v29; // [rsp+70h] [rbp-60h]
  unsigned __int64 v30; // [rsp+78h] [rbp-58h] BYREF
  unsigned int v31; // [rsp+80h] [rbp-50h]
  char v32; // [rsp+88h] [rbp-48h]
  char v33; // [rsp+90h] [rbp-40h]

  sub_30D08B0((__int64)&v16, (__int64)a3, *(_QWORD *)(a2 + 16), a2 + 80);
  v4 = *(_QWORD *)(a2 + 16);
  v5 = sub_B491C0((__int64)a3);
  v6 = sub_BC1CD0(v4, &unk_4F8FAE8, v5);
  v33 = 0;
  v7 = v6 + 8;
  if ( v24 )
  {
    v32 = 0;
    v25 = v16;
    v26 = v17;
    v27 = v18;
    if ( v23 )
    {
      v29 = v20;
      if ( v20 > 0x40 )
        sub_C43780((__int64)&v28, (const void **)&v19);
      else
        v28 = v19;
      v31 = v22;
      if ( v22 > 0x40 )
        sub_C43780((__int64)&v30, (const void **)&v21);
      else
        v30 = v21;
      v32 = 1;
    }
    v33 = 1;
  }
  v8 = sub_22077B0(0x98u);
  v9 = v8;
  if ( !v8 )
  {
    v15 = v33;
    goto LABEL_30;
  }
  sub_30CABE0(v8, a2, a3, v7, v33);
  *(_QWORD *)(v9 + 64) = a3;
  *(_BYTE *)(v9 + 136) = 0;
  v10 = v33 == 0;
  *(_QWORD *)v9 = &unk_4A32518;
  if ( v10 )
  {
    *(_BYTE *)(v9 + 144) = 1;
    goto LABEL_5;
  }
  v12 = v25;
  v10 = v32 == 0;
  *(_BYTE *)(v9 + 128) = 0;
  *(_QWORD *)(v9 + 72) = v12;
  *(_DWORD *)(v9 + 80) = v26;
  *(_QWORD *)(v9 + 88) = v27;
  if ( !v10 )
  {
    v13 = v29;
    *(_DWORD *)(v9 + 104) = v29;
    if ( v13 > 0x40 )
      sub_C43780(v9 + 96, (const void **)&v28);
    else
      *(_QWORD *)(v9 + 96) = v28;
    v14 = v31;
    *(_DWORD *)(v9 + 120) = v31;
    if ( v14 > 0x40 )
      sub_C43780(v9 + 112, (const void **)&v30);
    else
      *(_QWORD *)(v9 + 112) = v30;
    *(_BYTE *)(v9 + 128) = 1;
    v15 = v33;
    *(_BYTE *)(v9 + 136) = 1;
    *(_BYTE *)(v9 + 144) = 1;
LABEL_30:
    if ( !v15 )
      goto LABEL_5;
    goto LABEL_12;
  }
  *(_BYTE *)(v9 + 136) = 1;
  *(_BYTE *)(v9 + 144) = 1;
LABEL_12:
  v33 = 0;
  if ( v32 )
  {
    v32 = 0;
    if ( v31 > 0x40 && v30 )
      j_j___libc_free_0_0(v30);
    if ( v29 > 0x40 && v28 )
      j_j___libc_free_0_0(v28);
  }
LABEL_5:
  v10 = v24 == 0;
  *a1 = v9;
  if ( !v10 )
  {
    v24 = 0;
    if ( v23 )
    {
      v23 = 0;
      if ( v22 > 0x40 && v21 )
        j_j___libc_free_0_0(v21);
      if ( v20 > 0x40 && v19 )
        j_j___libc_free_0_0(v19);
    }
  }
  return a1;
}
