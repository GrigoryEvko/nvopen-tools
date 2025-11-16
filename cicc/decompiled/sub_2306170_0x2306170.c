// Function: sub_2306170
// Address: 0x2306170
//
_QWORD *__fastcall sub_2306170(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  _QWORD *v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  _QWORD *v13; // rbx
  __int64 v14; // rax
  __int64 v15; // rdi
  __int64 v16; // rsi
  __int64 v17; // rsi
  char *v19; // [rsp+0h] [rbp-F0h] BYREF
  unsigned int v20; // [rsp+8h] [rbp-E8h]
  char v21; // [rsp+10h] [rbp-E0h] BYREF
  __int64 v22; // [rsp+30h] [rbp-C0h]
  __int64 v23; // [rsp+38h] [rbp-B8h]
  __int64 v24; // [rsp+40h] [rbp-B0h]
  unsigned int v25; // [rsp+48h] [rbp-A8h]
  __int64 v26; // [rsp+50h] [rbp-A0h]
  int v27; // [rsp+58h] [rbp-98h]
  char *v28; // [rsp+60h] [rbp-90h] BYREF
  __int64 v29; // [rsp+68h] [rbp-88h]
  _BYTE v30[32]; // [rsp+70h] [rbp-80h] BYREF
  __int64 v31; // [rsp+90h] [rbp-60h]
  __int64 v32; // [rsp+98h] [rbp-58h]
  __int64 v33; // [rsp+A0h] [rbp-50h]
  unsigned int v34; // [rsp+A8h] [rbp-48h]
  __int64 v35; // [rsp+B0h] [rbp-40h]
  int v36; // [rsp+B8h] [rbp-38h]

  sub_22AB2A0((__int64)&v19, a2 + 8, a3, a4);
  v28 = v30;
  v29 = 0x100000000LL;
  if ( v20 )
    sub_23033F0((__int64)&v28, &v19, v20, v5, v6, v7);
  ++v22;
  v31 = 1;
  v32 = v23;
  v23 = 0;
  v33 = v24;
  v24 = 0;
  v34 = v25;
  v25 = 0;
  v35 = v26;
  v36 = v27;
  v8 = (_QWORD *)sub_22077B0(0x68u);
  v13 = v8;
  if ( v8 )
  {
    *v8 = &unk_4A0B420;
    v8[1] = v8 + 3;
    v8[2] = 0x100000000LL;
    if ( (_DWORD)v29 )
      sub_23033F0((__int64)(v8 + 1), &v28, v9, v10, v11, v12);
    v14 = v32;
    ++v31;
    v15 = 0;
    v16 = 0;
    v13[7] = 1;
    v13[8] = v14;
    v32 = 0;
    v13[9] = v33;
    v33 = 0;
    *((_DWORD *)v13 + 20) = v34;
    v34 = 0;
    v13[11] = v35;
    *((_DWORD *)v13 + 24) = v36;
  }
  else
  {
    v15 = v32;
    v16 = 16LL * v34;
  }
  sub_C7D6A0(v15, v16, 8);
  if ( v28 != v30 )
    _libc_free((unsigned __int64)v28);
  v17 = v25;
  *a1 = v13;
  sub_C7D6A0(v23, 16 * v17, 8);
  if ( v19 != &v21 )
    _libc_free((unsigned __int64)v19);
  return a1;
}
