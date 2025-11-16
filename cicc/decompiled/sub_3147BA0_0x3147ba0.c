// Function: sub_3147BA0
// Address: 0x3147ba0
//
__int64 *__fastcall sub_3147BA0(__int64 *a1, _QWORD *a2, __int64 a3)
{
  void (__fastcall *v4)(_BYTE *, __int64, __int64); // rax
  __int64 v6; // rax
  void (__fastcall *v7)(_BYTE *, _BYTE *, __int64); // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  unsigned __int64 v10; // r8
  __int64 v11; // r9
  unsigned __int64 v12; // rdx
  unsigned __int64 v13; // rax
  __int64 v14; // rsi
  __int64 v15; // rcx
  __int64 v16; // rdi
  unsigned __int64 v17; // r13
  unsigned __int64 v18; // r13
  unsigned __int64 v19; // rdi
  _BYTE v21[16]; // [rsp+0h] [rbp-B0h] BYREF
  void (__fastcall *v22)(_BYTE *, _BYTE *, __int64); // [rsp+10h] [rbp-A0h]
  __int64 v23; // [rsp+18h] [rbp-98h]
  __int64 v24; // [rsp+20h] [rbp-90h] BYREF
  char v25; // [rsp+28h] [rbp-88h]
  _BYTE v26[16]; // [rsp+30h] [rbp-80h] BYREF
  void (__fastcall *v27)(_BYTE *, _BYTE *, __int64); // [rsp+40h] [rbp-70h]
  __int64 v28; // [rsp+48h] [rbp-68h]
  unsigned __int64 v29; // [rsp+50h] [rbp-60h]
  unsigned __int64 v30; // [rsp+58h] [rbp-58h]
  __int64 v31; // [rsp+60h] [rbp-50h]
  __int64 v32; // [rsp+68h] [rbp-48h]
  __int64 v33; // [rsp+70h] [rbp-40h]
  unsigned int v34; // [rsp+78h] [rbp-38h]

  v4 = *(void (__fastcall **)(_BYTE *, __int64, __int64))(a3 + 16);
  v22 = 0;
  if ( !v4 )
  {
    v24 = 4;
    v25 = 1;
    v27 = 0;
    goto LABEL_18;
  }
  v4(v21, a3, 2);
  v6 = *(_QWORD *)(a3 + 24);
  v24 = 4;
  v25 = 1;
  v23 = v6;
  v7 = *(void (__fastcall **)(_BYTE *, _BYTE *, __int64))(a3 + 16);
  v27 = 0;
  v22 = v7;
  if ( !v7 )
  {
LABEL_18:
    v29 = 0;
    v30 = 0;
    v31 = 0;
    v32 = 0;
    v33 = 0;
    v34 = 0;
    goto LABEL_6;
  }
  v7(v26, v21, 2);
  v29 = 0;
  v30 = 0;
  v28 = v23;
  v31 = 0;
  v27 = v22;
  v32 = 0;
  v33 = 0;
  v34 = 0;
  if ( v22 )
  {
    sub_3145C60((__int64)&v24);
    if ( v22 )
      v22(v21, v21, 3);
  }
LABEL_6:
  if ( !sub_B2FC80((__int64)a2) )
    sub_3146A80(&v24, a2, v8, v9, v10, v11);
  v12 = v29;
  v13 = v30;
  v29 = 0;
  v14 = v34;
  v15 = v24;
  v30 = 0;
  a1[1] = v12;
  v16 = v32;
  *a1 = v15;
  a1[2] = v13;
  sub_C7D6A0(v16, 16 * v14, 8);
  v17 = v30;
  if ( v30 )
  {
    sub_C7D6A0(*(_QWORD *)(v30 + 8), 16LL * *(unsigned int *)(v30 + 24), 8);
    j_j___libc_free_0(v17);
  }
  v18 = v29;
  if ( v29 )
  {
    v19 = *(_QWORD *)(v29 + 32);
    if ( v19 != v29 + 48 )
      _libc_free(v19);
    sub_C7D6A0(*(_QWORD *)(v18 + 8), 8LL * *(unsigned int *)(v18 + 24), 4);
    j_j___libc_free_0(v18);
  }
  if ( v27 )
    v27(v26, v26, 3);
  return a1;
}
