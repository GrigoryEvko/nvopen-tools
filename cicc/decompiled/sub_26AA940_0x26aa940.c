// Function: sub_26AA940
// Address: 0x26aa940
//
__int64 __fastcall sub_26AA940(__int64 a1, __int64 a2)
{
  __int64 v3; // r13
  __int64 v5; // r15
  __int64 v6; // rsi
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rdx
  _QWORD *v10; // r14
  __int64 (__fastcall *v11)(__int64); // rax
  _BYTE *v12; // rdi
  __int64 (__fastcall *v13)(__int64); // rax
  char v14; // al
  __int64 v15; // rax
  unsigned __int64 *v16; // rcx
  __int64 v17; // rdx
  unsigned __int64 *v18; // r14
  unsigned int v19; // r12d
  unsigned __int8 *v21; // rsi
  __int64 v22; // [rsp-10h] [rbp-1F0h]
  __int64 v23; // [rsp+0h] [rbp-1E0h]
  unsigned __int64 *v24; // [rsp+8h] [rbp-1D8h]
  __int64 v25[4]; // [rsp+10h] [rbp-1D0h] BYREF
  _QWORD v26[7]; // [rsp+30h] [rbp-1B0h] BYREF
  unsigned int v27; // [rsp+68h] [rbp-178h]
  _QWORD *v28; // [rsp+70h] [rbp-170h]
  _QWORD v29[5]; // [rsp+80h] [rbp-160h] BYREF
  unsigned int v30; // [rsp+A8h] [rbp-138h]
  _QWORD *v31; // [rsp+B0h] [rbp-130h]
  _QWORD v32[5]; // [rsp+C0h] [rbp-120h] BYREF
  unsigned int v33; // [rsp+E8h] [rbp-F8h]
  char *v34; // [rsp+F0h] [rbp-F0h]
  char v35; // [rsp+100h] [rbp-E0h] BYREF
  __int64 (__fastcall **v36)(); // [rsp+120h] [rbp-C0h]
  __int64 v37; // [rsp+138h] [rbp-A8h]
  unsigned int v38; // [rsp+148h] [rbp-98h]
  _QWORD *v39; // [rsp+150h] [rbp-90h]
  _QWORD v40[5]; // [rsp+160h] [rbp-80h] BYREF
  unsigned int v41; // [rsp+188h] [rbp-58h]
  char *v42; // [rsp+190h] [rbp-50h]
  char v43; // [rsp+1A8h] [rbp-38h] BYREF

  v3 = a1 + 88;
  v5 = *(_QWORD *)(a2 + 208);
  sub_266FF60((__int64)v26, a1 + 88);
  v6 = *(_QWORD *)(a1 + 72);
  v25[0] = v5;
  v7 = *(_QWORD *)(a1 + 80);
  v25[1] = a2;
  v25[2] = a1;
  v25[3] = (__int64)v26;
  v8 = sub_251C7D0(a2, v6, v7, a1, 1, 0, 1);
  v9 = v22;
  if ( !v8 )
    goto LABEL_23;
  v10 = (_QWORD *)v8;
  v11 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v8 + 48LL);
  if ( v11 != sub_2534F50 )
  {
    v12 = (_BYTE *)((__int64 (__fastcall *)(_QWORD *, __int64, __int64))v11)(v10, v6, v22);
    v13 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v12 + 16LL);
    if ( v13 == sub_2505E30 )
      goto LABEL_4;
LABEL_26:
    v14 = ((__int64 (__fastcall *)(_BYTE *, __int64, __int64))v13)(v12, v6, v9);
    goto LABEL_5;
  }
  v12 = v10 + 11;
  v13 = *(__int64 (__fastcall **)(__int64))(v10[11] + 16LL);
  if ( v13 != sub_2505E30 )
    goto LABEL_26;
LABEL_4:
  v14 = v12[9];
LABEL_5:
  if ( v14 && !(*(unsigned __int8 (__fastcall **)(_QWORD *, __int64, __int64))(*v10 + 120LL))(v10, v6, v9) )
  {
    v15 = (*(__int64 (__fastcall **)(_QWORD *))(*v10 + 112LL))(v10);
    v16 = *(unsigned __int64 **)(v15 + 32);
    v23 = v15;
    v17 = *(unsigned int *)(v15 + 40);
    v24 = &v16[v17];
    if ( v24 != v16 )
    {
      v18 = *(unsigned __int64 **)(v15 + 32);
      while ( 1 )
      {
        sub_26A9FF0(v25, *v18, v17);
        if ( *(_BYTE *)(a1 + 96) )
          break;
        if ( v24 == ++v18 )
          break;
        LODWORD(v17) = *(_DWORD *)(v23 + 40);
      }
    }
    goto LABEL_12;
  }
LABEL_23:
  v21 = sub_250CBE0((__int64 *)(a1 + 72), v6);
  if ( v21 )
    sub_26A9FF0(v25, (unsigned __int64)v21, 1);
LABEL_12:
  v19 = (unsigned __int8)sub_266F260((__int64)v26, v3);
  v26[0] = off_49D3CA8;
  v40[0] = off_4A1FCF8;
  if ( v42 != &v43 )
    _libc_free((unsigned __int64)v42);
  sub_C7D6A0(v40[3], v41, 1);
  v36 = off_4A1FC98;
  if ( v39 != v40 )
    _libc_free((unsigned __int64)v39);
  sub_C7D6A0(v37, 8LL * v38, 8);
  v32[0] = off_4A1FC38;
  if ( v34 != &v35 )
    _libc_free((unsigned __int64)v34);
  sub_C7D6A0(v32[3], 8LL * v33, 8);
  v29[0] = off_4A1FBD8;
  if ( v31 != v32 )
    _libc_free((unsigned __int64)v31);
  sub_C7D6A0(v29[3], 8LL * v30, 8);
  v26[2] = off_4A1FB78;
  if ( v28 != v29 )
    _libc_free((unsigned __int64)v28);
  sub_C7D6A0(v26[5], 8LL * v27, 8);
  return v19;
}
