// Function: sub_2F38BD0
// Address: 0x2f38bd0
//
__int64 __fastcall sub_2F38BD0(__int64 a1, _QWORD *a2)
{
  __int64 v3; // rdi
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // r12
  __int64 v10; // rax
  __int64 v11; // r15
  __int64 v12; // rax
  __int64 v13; // r14
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // rdi
  unsigned int v20; // r12d
  __int64 v21; // rsi
  __int64 v23; // [rsp+0h] [rbp-F0h] BYREF
  __int64 v24; // [rsp+8h] [rbp-E8h]
  __int64 v25; // [rsp+10h] [rbp-E0h]
  __int64 v26; // [rsp+18h] [rbp-D8h]
  __int64 v27; // [rsp+20h] [rbp-D0h]
  __int64 v28; // [rsp+28h] [rbp-C8h]
  __int64 v29; // [rsp+30h] [rbp-C0h]
  __int64 v30; // [rsp+38h] [rbp-B8h]
  __int64 v31; // [rsp+40h] [rbp-B0h]
  __int64 v32; // [rsp+48h] [rbp-A8h]
  char *v33; // [rsp+50h] [rbp-A0h]
  __int64 v34; // [rsp+58h] [rbp-98h]
  __int64 v35; // [rsp+60h] [rbp-90h]
  char v36; // [rsp+68h] [rbp-88h] BYREF
  __int64 v37; // [rsp+88h] [rbp-68h]
  __int64 v38; // [rsp+90h] [rbp-60h]
  __int64 v39; // [rsp+98h] [rbp-58h]
  unsigned int v40; // [rsp+A0h] [rbp-50h]
  __int64 v41; // [rsp+A8h] [rbp-48h]
  __int64 v42; // [rsp+B0h] [rbp-40h]

  v41 = a1;
  v3 = *(_QWORD *)(a1 + 8);
  v23 = 0;
  v24 = 0;
  v25 = 0;
  v26 = 0;
  v27 = 0;
  v28 = 0;
  v29 = 0;
  v30 = 0;
  v31 = 0;
  v32 = 0;
  v33 = &v36;
  v34 = 4;
  LODWORD(v35) = 0;
  BYTE4(v35) = 1;
  v37 = 0;
  v38 = 0;
  v39 = 0;
  v40 = 0;
  v42 = 0;
  v4 = sub_B82360(v3, (__int64)&unk_501EB14);
  v9 = v4;
  if ( v4 )
    v9 = (*(__int64 (__fastcall **)(__int64, void *, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, char *, __int64, __int64))(*(_QWORD *)v4 + 104LL))(
           v4,
           &unk_501EB14,
           v5,
           v6,
           v7,
           v8,
           v23,
           v24,
           v25,
           v26,
           v27,
           v28,
           v29,
           v30,
           v31,
           v32,
           v33,
           v34,
           v35);
  v10 = sub_B82360(*(_QWORD *)(a1 + 8), (__int64)&unk_501EACC);
  v11 = v10;
  if ( v10 )
    v11 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v10 + 104LL))(v10, &unk_501EACC);
  v12 = sub_B82360(*(_QWORD *)(a1 + 8), (__int64)&unk_50208AC);
  v13 = v12;
  if ( v12 )
    v13 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v12 + 104LL))(v12, &unk_50208AC);
  v14 = sub_B82360(*(_QWORD *)(a1 + 8), (__int64)&unk_501FE44);
  v19 = v14;
  if ( v14 )
    v19 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v14 + 104LL))(v14, &unk_501FE44);
  if ( v9 )
    v9 += 200;
  if ( v11 )
    v11 += 200;
  if ( v13 )
    v13 += 200;
  v24 = v9;
  if ( v19 )
    v19 += 200;
  v25 = v11;
  v26 = v13;
  v27 = v19;
  v20 = sub_2F36310((__int64)&v23, a2, v15, v16, v17, v18);
  if ( v40 )
    v21 = 16LL * v40;
  else
    v21 = 0;
  sub_C7D6A0(v38, v21, 8);
  if ( !BYTE4(v35) )
    _libc_free((unsigned __int64)v33);
  sub_C7D6A0(v29, 12LL * (unsigned int)v31, 4);
  return v20;
}
