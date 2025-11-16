// Function: sub_ABCC80
// Address: 0xabcc80
//
__int64 __fastcall sub_ABCC80(__int64 a1, __int64 a2, char a3)
{
  char v5; // al
  unsigned int v6; // esi
  unsigned int v8; // r13d
  unsigned int v9; // eax
  bool v10; // r15
  unsigned int v11; // eax
  unsigned int v12; // eax
  unsigned int v13; // eax
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // rax
  unsigned int v16; // eax
  __int64 v17; // rdx
  unsigned int v18; // esi
  unsigned int v19; // eax
  int v20; // eax
  unsigned int v21; // eax
  unsigned int v22; // eax
  int v23; // eax
  bool v24; // [rsp+8h] [rbp-A8h]
  __int64 v25; // [rsp+10h] [rbp-A0h] BYREF
  unsigned int v26; // [rsp+18h] [rbp-98h]
  unsigned __int64 v27; // [rsp+20h] [rbp-90h] BYREF
  unsigned int v28; // [rsp+28h] [rbp-88h]
  unsigned __int64 v29; // [rsp+30h] [rbp-80h] BYREF
  unsigned int v30; // [rsp+38h] [rbp-78h]
  __int64 v31; // [rsp+40h] [rbp-70h] BYREF
  unsigned int v32; // [rsp+48h] [rbp-68h]
  unsigned __int64 v33; // [rsp+50h] [rbp-60h] BYREF
  unsigned int v34; // [rsp+58h] [rbp-58h]
  unsigned __int64 v35; // [rsp+60h] [rbp-50h] BYREF
  unsigned int v36; // [rsp+68h] [rbp-48h]
  unsigned __int64 v37; // [rsp+70h] [rbp-40h] BYREF
  unsigned int v38; // [rsp+78h] [rbp-38h]

  v5 = sub_AAF7D0(a2);
  v6 = *(_DWORD *)(a2 + 8);
  if ( v5 )
  {
    sub_AADB10(a1, v6, 0);
    return a1;
  }
  sub_9691E0((__int64)&v25, v6, 0, 0, 0);
  if ( !a3 || !sub_AB1B10(a2, (__int64)&v25) )
  {
    sub_AB0A00((__int64)&v33, a2);
    v11 = v34;
    if ( v34 <= 0x40 )
    {
      if ( v33 )
      {
        _BitScanReverse64(&v15, v33);
        v11 = v34 - 64 + (v15 ^ 0x3F);
      }
    }
    else
    {
      v11 = sub_C444A0(&v33);
    }
    sub_9691E0((__int64)&v35, *(_DWORD *)(a2 + 8), v11, 0, 0);
    sub_C46A40(&v35, 1);
    v12 = v36;
    v36 = 0;
    v38 = v12;
    v37 = v35;
    sub_AB0910((__int64)&v29, a2);
    v13 = v30;
    if ( v30 > 0x40 )
    {
      v13 = sub_C444A0(&v29);
    }
    else if ( v29 )
    {
      _BitScanReverse64(&v14, v29);
      v13 = v30 - 64 + (v14 ^ 0x3F);
    }
    sub_9691E0((__int64)&v31, *(_DWORD *)(a2 + 8), v13, 0, 0);
    sub_9875E0(a1, &v31, (__int64 *)&v37);
    if ( v32 > 0x40 && v31 )
      j_j___libc_free_0_0(v31);
    if ( v30 > 0x40 && v29 )
      j_j___libc_free_0_0(v29);
    if ( v38 > 0x40 && v37 )
      j_j___libc_free_0_0(v37);
    if ( v36 > 0x40 && v35 )
      j_j___libc_free_0_0(v35);
    if ( v34 > 0x40 && v33 )
      j_j___libc_free_0_0(v33);
    goto LABEL_30;
  }
  v8 = *(_DWORD *)(a2 + 8);
  if ( v8 > 0x40 )
  {
    if ( v8 == (unsigned int)sub_C444A0(a2) )
      goto LABEL_8;
LABEL_36:
    sub_9865C0((__int64)&v35, a2 + 16);
    sub_C46F20(&v35, 1);
    v16 = v36;
    v36 = 0;
    v38 = v16;
    v37 = v35;
    v24 = sub_9867B0((__int64)&v37);
    sub_969240((__int64 *)&v37);
    sub_969240((__int64 *)&v35);
    if ( v24 )
    {
      v23 = sub_9871A0(a2);
      v18 = *(_DWORD *)(a2 + 8);
      v17 = (unsigned int)(v23 + 1);
    }
    else
    {
      v17 = *(unsigned int *)(a2 + 8);
      v18 = *(_DWORD *)(a2 + 8);
    }
    sub_9691E0((__int64)&v35, v18, v17, 0, 0);
    sub_9865C0((__int64)&v37, (__int64)&v25);
    sub_AADC30(a1, (__int64)&v37, (__int64 *)&v35);
    sub_969240((__int64 *)&v37);
    sub_969240((__int64 *)&v35);
    goto LABEL_30;
  }
  if ( *(_QWORD *)a2 )
    goto LABEL_36;
LABEL_8:
  sub_9865C0((__int64)&v35, a2 + 16);
  sub_C46F20(&v35, 1);
  v9 = v36;
  v36 = 0;
  v38 = v9;
  v37 = v35;
  v10 = sub_9867B0((__int64)&v37);
  sub_969240((__int64 *)&v37);
  sub_969240((__int64 *)&v35);
  if ( v10 )
  {
    sub_AADB10(a1, *(_DWORD *)(a2 + 8), 0);
  }
  else
  {
    sub_9865C0((__int64)&v33, a2);
    sub_C46A40(&v33, 1);
    v19 = v34;
    v34 = 0;
    v36 = v19;
    v35 = v33;
    v20 = sub_9871A0((__int64)&v35);
    sub_9691E0((__int64)&v37, *(_DWORD *)(a2 + 8), (unsigned int)(v20 + 1), 0, 0);
    sub_9865C0((__int64)&v27, a2 + 16);
    sub_C46F20(&v27, 1);
    v21 = v28;
    v28 = 0;
    v30 = v21;
    v29 = v27;
    v22 = sub_9871A0((__int64)&v29);
    sub_9691E0((__int64)&v31, *(_DWORD *)(a2 + 8), v22, 0, 0);
    sub_AADC30(a1, (__int64)&v31, (__int64 *)&v37);
    sub_969240(&v31);
    sub_969240((__int64 *)&v29);
    sub_969240((__int64 *)&v27);
    sub_969240((__int64 *)&v37);
    sub_969240((__int64 *)&v35);
    sub_969240((__int64 *)&v33);
  }
LABEL_30:
  if ( v26 > 0x40 && v25 )
    j_j___libc_free_0_0(v25);
  return a1;
}
